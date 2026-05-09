# End-to-End Smoke Test

> Faz 1 platformunun çalışmasını doğrulamak için adım adım test senaryosu.
>
> **Önkoşul:** `docker compose --profile infra up -d` ile altyapı ayağa kalktı +
> `docker compose up -d --build` ile tüm servisler build edildi.

## Test Akışı (Mutlu Yol)

```
[1] register   → identity-service: yeni user + JWT token
[2] login      → JWT (re-fetch — register zaten döndü ama test ettik)
[3] /chats     → conversation-service: yeni chat oluştur
[4] /messages  → conversation-service: user message + Kafka event publish
[5] /inference → inference-orchestrator: SSE stream başlar
                 ├─ guardrail check (bypass — Faz 1)
                 ├─ quota check (bypass — Faz 1)
                 ├─ routing (hardcode goktug-medium)
                 └─ inference-worker-goktug: token-by-token SSE
[6] /chats/{id}/messages → conversation-service: assistant message persist
                            edilmiş mi (saga consumer test)
[7] /chats/search?q=     → ES projector test
```

## 0. Sağlık Kontrolü

```bash
# API Gateway
curl -s http://localhost:8080/actuator/health | jq

# Identity service (gateway'siz direkt — internal)
docker compose exec identity-service curl -s http://localhost:8081/actuator/health

# Inference worker
curl -s http://localhost:9001/v1/health | jq

# Keycloak — goktuggpt realm var mı?
curl -s http://localhost:8180/realms/goktuggpt/.well-known/openid-configuration | jq .issuer
```

Beklenen: hepsi `UP` veya `healthy`. Inference worker `model_loaded: true` döner
(checkpoints/best_model.pt + tokenizer.json mount edilmişse).

---

## 1. Register

```bash
curl -s -X POST http://localhost:8080/api/v1/auth/register \
  -H 'Content-Type: application/json' \
  -d '{
    "email": "test@goktug.dev",
    "password": "TestPass1234",
    "displayName": "Test User"
  }' | jq
```

Beklenen response (201 Created):
```json
{
  "accessToken": "eyJ...",
  "refreshToken": "eyJ...",
  "expiresIn": 1800,
  "userId": "550e8400-e29b-41d4-a716-446655440000",
  "email": "test@goktug.dev"
}
```

`accessToken`'ı kaydet:
```bash
TOKEN=$(curl -s -X POST http://localhost:8080/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"test@goktug.dev","password":"TestPass1234"}' | jq -r .accessToken)
echo $TOKEN
```

---

## 2. Yeni Chat Oluştur

```bash
CHAT=$(curl -s -X POST http://localhost:8080/api/v1/chats \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"title":"Smoke test chat"}')
echo $CHAT | jq

CHAT_ID=$(echo $CHAT | jq -r .id)
echo "CHAT_ID=$CHAT_ID"
```

Beklenen: `{ "id": "...", "title": "Smoke test chat", ... }`.

`chat.created.v1` event Kafka'ya basıldı. Doğrula:
```bash
# Kafka UI'da: http://localhost:8090 → topic: chat.events → message gör
```

ES projector test (eventually consistent — birkaç saniye bekle):
```bash
curl -s "http://localhost:8080/api/v1/chats/search?q=Smoke" \
  -H "Authorization: Bearer $TOKEN" | jq
```
→ Yeni chat gelmeli.

---

## 3. User Message Gönder

```bash
IDEMP_KEY=$(uuidgen || echo "test-$(date +%s)")

MSG=$(curl -s -X POST "http://localhost:8080/api/v1/chats/$CHAT_ID/messages" \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Idempotency-Key: $IDEMP_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"content":"merhaba dünya, sen kimsin?"}')
echo $MSG | jq

MSG_ID=$(echo $MSG | jq -r .id)
```

Beklenen: 201 Created, message persist edildi.

`message.user-sent.v1` event Kafka `message.events` topic'ine düştü
(OutboxPoller 1s'de basar).

**Idempotency test** — aynı key ile yeniden POST at:
```bash
curl -s -X POST "http://localhost:8080/api/v1/chats/$CHAT_ID/messages" \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Idempotency-Key: $IDEMP_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"content":"merhaba dünya, sen kimsin?"}' | jq .id
```
→ Aynı `id` dönmeli (yeni message yaratılmadı).

---

## 4. Inference Stream

```bash
curl -N -X POST http://localhost:8080/api/v1/inference/stream \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d "{
    \"chatId\":\"$CHAT_ID\",
    \"userMessageId\":\"$MSG_ID\",
    \"text\":\"merhaba dünya, sen kimsin?\"
  }"
```

Beklenen SSE stream:
```
event:token
data:{"jobId":"...","type":"token","content":"Merhaba "}

event:token
data:{"jobId":"...","type":"token","content":"ben "}

event:token
data:{"jobId":"...","type":"token","content":"GoktugGPT"}
...
```

`-N` flag'i `curl`'ın buffer'ı disable etmesini sağlar (SSE için kritik).

---

## 5. Assistant Message Persist Edildi mi?

Stream tamamlanınca `inference.completed.v1` event Kafka'ya gider →
`InferenceCompletedConsumer` (conversation-service) yakalar → `messages`
tablosuna assistant row insert eder.

```bash
sleep 2
curl -s "http://localhost:8080/api/v1/chats/$CHAT_ID/messages" \
  -H "Authorization: Bearer $TOKEN" | jq '.[] | {sender, content: (.content[:80])}'
```

Beklenen 2 row:
```json
{ "sender": "USER", "content": "merhaba dünya, sen kimsin?" }
{ "sender": "ASSISTANT", "content": "Merhaba ben GoktugGPT..." }
```

---

## 6. Distributed Trace

Jaeger UI: http://localhost:16686

- Service: `api-gateway` seç → recent traces
- Tek bir trace: `api-gateway` → `inference-orchestrator` → `inference-worker-goktug`
  hatları görünmeli (4-5 span)
- Kafka producer/consumer span'ları var mı? (`inference.events` publish + consume)

---

## Failure Senaryoları

### A. Quota Exceeded (Faz 2 — billing enabled iken)
`billing.enabled=true` ile orchestrator `BillingClient.checkQuota()` 0 döndürürse:
- SSE response: `{"type":"error","content":"Quota exceeded","terminalState":"QUOTA_EXCEEDED"}`
- inference_jobs tablosunda state = QUOTA_EXCEEDED

### B. Worker Down
`docker compose stop inference-worker-goktug` → POST /inference/stream:
- Resilience4j circuit breaker fail rate threshold'u aşınca → CircuitBreaker OPEN
- SSE response: `{"type":"error","content":"...","terminalState":"INFERENCE_FAILED"}`

### C. Saga Recovery
1. Inference başlat
2. Hızlıca `docker compose stop inference-orchestrator`
3. inference_jobs tablosunda STREAMING state'inde job kalır
4. `docker compose start inference-orchestrator` (10 dk bekleyip ya da
   `saga.recovery.stale-after-minutes=0` set ederek hızlandır)
5. SagaRecoveryWorker job'u INFERENCE_FAILED'a çeker, refund tetiklenir

---

## Debugging İpuçları

| Sorun | Çözüm |
|-------|-------|
| 401 Unauthorized | Token expired (1800s) — re-login |
| 503 from worker | `docker compose logs inference-worker-goktug` — model load ettiği yerde NoSuchKeyException? Checkpoint mount edilmedi |
| ES projector çalışmıyor | `docker compose logs conversation-service` — Elasticsearch connect timeout? |
| Outbox Kafka'ya basılmıyor | conversation-service log: `Outbox: publishing N events` mesajı geliyor mu? OutboxPoller @Scheduled aktif mi (`@EnableScheduling`)? |
| Saga recovery hiç çalışmıyor | `saga.recovery.interval-ms` ve `stale-after-minutes` config kontrol et |
| Trace Jaeger'da yok | Tüm servisler `OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317` görüyor mu? Agent JAR `-javaagent` flag'inde var mı? |

# inference-orchestrator

> **Saga Orchestrator** — kullanıcı mesajını AI cevabına dönüştüren çok adımlı dağıtık işlemi yönetir.

## Sorumluluklar

- **Saga akışı:** guardrail → quota → routing → streaming → completion
- **SSE relay:** worker'dan gelen token'ları (gRPC stream veya SSE) frontend'e proxy
- **Compensating transactions:** failure'da quota'yı geri ver, message'ı failed işaretle
- **Saga state persistence:** orchestrator restart olsa bile recovery
- **Resilience patterns:** Circuit Breaker, Retry, Timeout, Bulkhead — her downstream için

## Mimari Karar — Orchestration vs Choreography

**Choreography:** Servisler birbirini event ile tetikler. Avantaj: gevşek bağlılık. Dezavantaj: akışı izlemek zor, "şu an hangi adımdayız?" sorusunun cevabı yok.

**Orchestration:** Merkezi bir koordinatör (bu servis) tüm adımları yönetir. Avantaj: deterministic, observable, debugging kolay. Dezavantaj: orchestrator SPOF, ama replication ile aşılır.

→ **Bu projede orchestration seçildi** çünkü streaming akışı + compensation logic merkezi olunca daha temiz.

## Saga Adımları

```
USER MESSAGE
    │
    ▼
[1] GUARDRAIL_CHECK    → guardrail-service
    │  fail → BLOCKED  → user'a "this prompt was blocked" döner
    ▼
[2] QUOTA_CHECK        → billing-service
    │  fail → QUOTA_EXCEEDED → 429 döner
    ▼
[3] ROUTING            → router-service (hangi modele git?)
    │
    ▼
[4] STREAMING          → inference-worker-{goktug|external}
    │  token-by-token Frontend'e SSE
    │  fail → INFERENCE_FAILED → COMPENSATE: quota refund
    ▼
[5] COMPLETED
    └─ inference.completed.v1 publish
       (consumer: conversation-service → assistant message persist
                  billing-service → token usage düş
                  telemetry-consumer → RLHF data)
```

## Endpoints

| Method | Path | Açıklama |
|--------|------|----------|
| POST | /api/v1/inference/stream | SSE — body: `{chatId, userMessageId, text, modelHint?}` |
| GET | /api/v1/inference/jobs/{jobId} | Saga state sorgulama |

## Eventler

**Publish:**
- `inference.started.v1`
- `inference.completed.v1`
- `inference.failed.v1`

**Subscribe:**
- (yok — orchestrator sync stream üzerinden çalışır)

## Resilience Konfigleri

`application.yml`'de her downstream için ayrı CB + Bulkhead:
- `guardrail`: 2s timeout, 50% fail rate → open, 10s wait
- `worker-goktug`: 30s timeout (streaming), 60% fail rate → open, 30s wait, 100 concurrent
- `billing`: 3 retry, exponential backoff (100ms → 200ms → 400ms)

## Port

`8083` (internal — gateway pass-through)

## TODO (bir sonraki session)

1. **Domain entity'leri:** `InferenceJob`, `SagaStep`, `JobRepository`, `StepRepository`
2. **Downstream client'ları (WebClient based):**
   - `GuardrailClient` (POST /v1/check)
   - `QuotaClient` (GET /api/v1/billing/me/quota)
   - `RouterClient` (POST /v1/route)
   - `InferenceWorkerClient` (SSE GET /v1/generate)
3. **InferenceController:** `POST /api/v1/inference/stream` — `Flux<ServerSentEvent>` döner
4. **InferenceSaga.execute()**'i tamamla — yorumdaki chain'i implement et
5. **Compensating actions:** `BillingClient.refund(jobId)`, persist failed state
6. **Recovery worker:** `@Scheduled` — `state IN ('STREAMING', 'ROUTING', ...)` and `updated_at < now() - 10min` olan jobları cleanup et (orchestrator crash recovery)
7. **`InferenceEventPublisher`:** completed/failed event'leri Kafka'ya bas
8. **End-to-end test:** mock guardrail + mock worker ile saga flow happy path + failure path

## Çalıştırma

```bash
docker compose --profile infra up -d
mvn -pl services/inference-orchestrator -am spring-boot:run
```

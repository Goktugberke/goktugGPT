# goktugGPT Platform

> ChatGPT/Gemini/Claude tarzı bir AI SaaS platformu — GoktugGPT modelinin üzerine mikroservis mimarisinde inşa edildi.

## Hızlı Başlangıç

### 1. Sadece altyapıyı kaldır (henüz servisleri build etmeden)

```bash
docker compose --profile infra up -d
```

Açılan UI'lar:
- Keycloak: http://localhost:8180 (admin/admin)
- MinIO Console: http://localhost:9090 (minio/minio12345)
- Kafka UI: http://localhost:8090
- Jaeger: http://localhost:16686
- Grafana: http://localhost:3000 (admin/admin)
- Kibana: http://localhost:5601

### 2. Tüm servisleri kaldır (servis kodu hazır olduktan sonra)

```bash
docker compose up -d --build
```

### 3. Sadece tek bir servisi çalıştır (geliştirme)

```bash
# Önce infra
docker compose --profile infra up -d

# Sonra sadece o servisi (Maven ile lokal)
mvn -pl services/conversation-service -am spring-boot:run
```

## Yapı

```
platform/
├── MICROSERVICES_PLAN.md    ← MASTER PLAN (en önce bunu oku)
├── PHASE2_TODO.md           ← faz 2 listesi
├── PHASE3_TODO.md           ← faz 3 listesi
├── docker-compose.yml       ← tüm infra + servisler
├── pom.xml                  ← Maven parent (Java servisler için)
├── infra/                   ← Keycloak realm, Prometheus config, K8s/Helm (TBD)
├── shared-contracts/
│   └── events/              ← JSON Schema event tanımları (kritik!)
├── libs/                    ← ortak Java/Python kütüphaneleri (TBD)
└── services/                ← her bir mikroservis
    ├── api-gateway/
    ├── identity-service/
    ├── conversation-service/   ← Outbox + Idempotency reference impl
    ├── inference-orchestrator/ ← Saga reference impl
    ├── inference-worker-goktug/
    ├── asset-service/
    ├── guardrail-service/      [Faz 2]
    └── billing-service/        [Faz 2]
```

## Servis Listesi (özet)

| Servis | Stack | Port | Faz |
|--------|-------|:----:|:---:|
| api-gateway | Java/WebFlux | 8080 | 1 |
| identity-service | Java + Keycloak | 8081 | 1 |
| conversation-service | Java + PG | 8082 | 1 |
| inference-orchestrator | Java/WebFlux | 8083 | 1 |
| inference-worker-goktug | Python/FastAPI | 9001 | 1 |
| asset-service | Java + MinIO | 8084 | 1 |
| guardrail-service | Python | 9003 | 2 |
| router-service | Python | 9002 | 2 |
| billing-service | Java | 8085 | 2 |
| telemetry-consumer | Java/Kafka | 8086 | 2 |
| notification-service | Java/WS | 8087 | 2 |
| rag-service | Python | 9004 | 3 |
| tool-service | Node | 7001 | 3 |
| audit-service | Java | 8088 | 3 |
| feature-flag-service | Java | 8089 | 3 |

## Devam Etmek İçin

**Bir sonraki session'da:** `MICROSERVICES_PLAN.md` → "Bölüm 6 — İlerleme Durumu" altındaki TODO listesi sıradaki adımları gösterir.

Hızlı yol haritası:
1. `identity-service` JPA entity + Keycloak admin client
2. `conversation-service` REST controller + service layer (Outbox/Idempotency zaten yazıldı)
3. `inference-orchestrator` Saga implement (downstream WebClient'lar dahil)
4. `inference-worker-goktug` modeli gerçekten yükle
5. End-to-end smoke test: register → login → chat → message → SSE token akışı

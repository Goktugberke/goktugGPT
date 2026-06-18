# Phase 2 — Production-Ready Features

> Faz 1 (MVP) çalıştığını doğruladıktan sonra bu listeyi sırayla yap.

## Durum (güncel)

| # | Madde | Durum |
|---|-------|-------|
| 2.1–2.5 | guardrail, router, billing, telemetry, notification servisleri | ✅ |
| 2.6 | OpenTelemetry → Jaeger tracing | ✅ (8/8 servis, Saga tek trace) |
| 2.7 | Spring Cloud Config | ✅ (Git backend + seed) |
| 2.8 | Service Mesh (Istio) | 📘 Kılavuz hazır (`docs/guides/ISTIO_SETUP.md`) — kullanıcı yapacak |
| 2.9 | CQRS chat search (Elasticsearch) | ✅ |
| 2.10 | Schema Registry (Apicurio) | ✅ (8 JSON Schema + backward-compat) |
| 2.11 | Helm Charts | ✅ (umbrella chart, dev/prod values) |
| 2.12 | GitHub Actions CI/CD | 📘 Kılavuz hazır (`docs/guides/GITHUB_ACTIONS_CICD.md`) — kullanıcı yapacak |
| 2.13 | Testcontainers integration tests | ⏳ Ertelendi (indirme/bant genişliği) |
| 2.14 | Load testing (k6/Gatling) | ⏳ Ertelendi |
| 2.15 | React frontend | ⏳ Ayrı `frontend/` üst-dizininde, tasarım aşamasında |

> Aşağısı orijinal detaylı planlardır (referans için korunuyor).

## Servisler

### 2.1 guardrail-service (Python)
- Bkz [`services/guardrail-service/README.md`](services/guardrail-service/README.md)
- DeBERTa prompt injection + Detoxify toxicity + PII regex
- Inference-orchestrator saga'sının ilk adımı

### 2.2 router-service (Python)
- Embedding-based prompt classifier
- Çıktı: `{model: "goktug-mini" | "goktug-pro" | "external-claude"}`
- Hot path — <50ms hedef

### 2.3 billing-service (Java)
- Bkz [`services/billing-service/README.md`](services/billing-service/README.md)
- Redis Token Bucket (Lua), usage tracking, free plan auto-create

### 2.4 telemetry-consumer (Java)
- Tüm `*.v1` eventleri consume → Elasticsearch (analytics) + S3/DLS (RLHF cold storage)
- Kibana dashboards: latency, error rate, token usage, popular models

### 2.5 notification-service (Java + WebSocket)
- WebSocket endpoint `/ws/notifications`
- Email gönderimi (SendGrid/SMTP)
- Trigger: `user.registered.v1` (welcome), `quota.exceeded.v1` (upgrade), async job complete

## Cross-cutting

### 2.6 OpenTelemetry → Jaeger Distributed Tracing
- Her servise `opentelemetry-spring-boot-starter` ekle
- `traceparent` header api-gateway → tüm downstream'lere propagate
- Saga akışını Jaeger'da tek trace olarak gör

### 2.7 Spring Cloud Config (config server)
- Tüm servislerin `application.yml`'ini Git repo'dan okuması
- Dynamic refresh (`@RefreshScope`)
- Sensitive değerler Vault'tan (Faz 3)

### 2.8 Service Mesh (Istio) — opsiyonel
- mTLS otomatik
- Traffic split (canary deployment)
- Sidecar proxy ile hot reload

### 2.9 CQRS Read Model — Conversation Search
- `conversation-service` → `chat.created.v1`, `chat.title-changed.v1` event publish
- **search-projector** servisi (yeni veya `conversation-service` içinde) → Elasticsearch index
- `GET /api/v1/chats/search?q=` ES'e gider
- Reindex job: full reindex için `conversation-service` initial state replay

### 2.10 Schema Registry (Confluent veya Apicurio)
- JSON Schema → registry'de versiyonlu
- Producer + Consumer compile-time'da schema'ya bağlı
- Backward compatibility check

## Operasyonel

### 2.11 Helm Charts
- `infra/helm/<service>/` — her servis için chart
- `values-dev.yaml` / `values-prod.yaml`
- Resource limits, HPA (HorizontalPodAutoscaler), liveness/readiness probes

### 2.12 GitHub Actions CI/CD
- Per-service workflow: build + test + Docker push (GHCR)
- Path filter (`paths:` in workflow) — sadece değişen servis build edilir
- ArgoCD ile K8s'e deploy

### 2.13 Testcontainers Integration Tests
- Her servis için `@Testcontainers` (Postgres + Kafka)
- `OutboxPoller` flow E2E test
- Saga happy path + her failure path

### 2.14 Load Testing (k6 veya Gatling)
- 100 concurrent SSE stream
- p50, p95, p99 latency budgets
- Saga step timing breakdown

## Frontend (Faz 2'de başlanır)

### 2.15 React + Vite + TypeScript Frontend
- Login/Register (Keycloak Adapter)
- Chat list + active chat
- SSE stream consumer
- File upload (presigned URL flow)
- Custom instructions UI
- Dark mode

> Frontend planı ayrı `FRONTEND_PLAN.md`'de detaylanacak (kullanıcı tasarım kararını verince).

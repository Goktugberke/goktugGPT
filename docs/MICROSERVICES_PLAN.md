# goktugGPT Platform — Microservices Master Plan

> **Amaç:** goktugGPT modelini (yerel transformer LLM) üreten/serve eden, kullanıcıların hesap açıp sohbet edebildiği, ChatGPT / Gemini / Claude tarzı bir SaaS platformunu **mikroservis mimarisinde** baştan tasarlamak.
>
> **Bağlam:** Eski monolit `4grands/saritaygpt/backend` (Spring Boot tek modül) ve `agent-service` (NestJS) yapısı amatör seviyede coupling içeriyor — `ChatService` + `AgentGateway` + `AzureDataLakeService` aynı thread'de file upload + chat persist + agent call + DB write yapıyor. Yeni mimari bunu **DDD bounded context** sınırlarıyla parçalara böler.
>
> **Bu döküman canlıdır.** Her faz tamamlandıkça güncellenir. Bir sonraki session bunu okuyarak nereden devam edeceğini bilir.

---

## 0. Yönetici Özeti

- **14 mikroservis** + **1 API Gateway** + **1 IAM (Keycloak)**.
- 3 fazlı geliştirme: **Faz 1 (MVP — 6 servis)**, **Faz 2 (özellikler — 5 servis)**, **Faz 3 (kurumsal — 4 servis)**.
- **Polyglot:** Java 21 + Spring Boot 3 (iş mantığı), Python 3.11 + FastAPI (AI/ML), Node.js (MCP tool bridge), TypeScript (frontend — ayrı plan).
- **Event-driven backbone:** Apache Kafka.
- **Sync iletişim:** REST (dış) + gRPC (iç, yüksek throughput) + SSE/WebSocket (streaming).
- **Tek ortak DB yok:** her servis kendi DB'sine sahip (Database-per-Service).
- **Distributed transaction:** Saga (Orchestration tabanlı) + Transactional Outbox + Idempotency.
- **Resilience:** Resilience4j (Circuit Breaker, Retry, Bulkhead, Timeout).
- **Observability:** OpenTelemetry → Jaeger (tracing) + Prometheus (metrics) + ELK (logs).

---

## 1. Domain Analizi (DDD)

### 1.1 Bounded Contexts

Sistem 10 domain alanına bölünür:

| # | Bounded Context | Sorumluluk | Aggregate Roots |
|---|----------------|-----------|-----------------|
| 1 | **Identity & Access** | Kayıt, login, JWT, profil, custom instructions | User, Profile |
| 2 | **Conversation** | Sohbet ve mesaj CRUD, geçmiş, arama | Chat, Message |
| 3 | **AI Inference** | Model serving, streaming, model routing | InferenceJob |
| 4 | **Asset Management** | Dosya upload, blob storage, presigned URL | Asset |
| 5 | **Safety & Guardrail** | Prompt injection, toksisite, PII tarama | GuardrailReport |
| 6 | **Knowledge / RAG** | Vektör DB, retrieval, embedding | Document, Chunk |
| 7 | **Tool Orchestration** | MCP, web search, kod çalıştırma | ToolInvocation |
| 8 | **Billing & Quota** | Token sayımı, abonelik, rate limit | UsageEvent, Subscription |
| 9 | **Telemetry & Audit** | Loglar, RLHF feedback, denetim | TelemetryEvent, AuditRecord |
| 10 | **Notification** | Push, email, in-app bildirimler | Notification |

### 1.2 Servis Haritası

```
                          ┌──────────────────────┐
                          │   Web / Mobile UI    │
                          └──────────┬───────────┘
                                     │ HTTPS / WSS
                          ┌──────────▼───────────┐
                          │   API Gateway        │  (Spring Cloud Gateway)
                          │   - Routing          │
                          │   - JWT validation   │
                          │   - Rate limit       │
                          │   - SSE/WS bridge    │
                          └──────────┬───────────┘
                                     │
        ┌───────────┬────────────────┼───────────────┬──────────────┐
        ▼           ▼                ▼               ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────┐ ┌──────────┐
│  identity    │ │ conversation │ │  inference   │ │   asset    │ │ billing  │
│  service     │ │  service     │ │  orchestr.   │ │  service   │ │ service  │
│  (Java)      │ │  (Java)      │ │  (Java/Webfx)│ │  (Java)    │ │  (Java)  │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └─────┬──────┘ └────┬─────┘
       │                │                │                │              │
   ┌───▼───┐        ┌───▼───┐        ┌───▼─────────────┐  │              │
   │Keycloak│       │Postgres│       │ ┌──────────────┐│  │              │
   │  +     │       │  +     │       │ │  router-svc  ││  │              │
   │Postgres│       │ Outbox │       │ │  (Python)    ││  │              │
   └────────┘       └────────┘       │ └──────┬───────┘│  │              │
                                     │        │        │  │              │
                                     │ ┌──────▼───────┐│  │              │
                                     │ │ guardrail-svc││  │              │
                                     │ │  (Python)    ││  │              │
                                     │ └──────┬───────┘│  │              │
                                     │        │        │  │              │
                                     │ ┌──────▼───────┐│  │              │
                                     │ │ inf-worker-  ││  │              │
                                     │ │   goktug     ││  │              │
                                     │ │  (Python)    ││  │              │
                                     │ └──────────────┘│  │              │
                                     │ ┌──────────────┐│  │              │
                                     │ │ inf-worker-  ││  │              │
                                     │ │  external    ││  │              │
                                     │ │ (OpenAI etc) ││  │              │
                                     │ └──────────────┘│  │              │
                                     └─────────────────┘  │              │
                                                          ▼              ▼
                                                       ┌──────┐    ┌────────┐
                                                       │MinIO │    │Postgres│
                                                       └──────┘    └────────┘

       ═══════════════════════════════════════════════════════════════
                     Kafka (Event Bus) — async backbone
       ═══════════════════════════════════════════════════════════════
              ▲              ▲             ▲             ▲
              │              │             │             │
       ┌──────┴──────┐  ┌────┴─────┐ ┌────┴─────┐ ┌────┴───────┐
       │ telemetry   │  │ rag      │ │ tool     │ │notification│
       │ consumer    │  │ service  │ │ service  │ │  service   │
       │ (Java)      │  │ (Python) │ │ (Node)   │ │  (Java)    │
       └──────┬──────┘  └────┬─────┘ └──────────┘ └────────────┘
              ▼              ▼
        ┌──────────┐    ┌─────────┐
        │Elasticsr.│    │pgvector │
        │  + DLS   │    │  /Milvus│
        └──────────┘    └─────────┘
```

---

## 2. Servisler — Tam Spec

> Her servis için: **dil/framework**, **port**, **DB**, **publish ettiği eventler**, **subscribe olduğu eventler**, **REST endpointler**, **kullanılan patternler**, **faz**.

### Faz 1 — MVP (Çalışan Bir Chat Platformu)

#### 2.1 `api-gateway` — Edge Streaming Gateway
- **Stack:** Spring Cloud Gateway (Reactive) + Java 21
- **Port:** 8080 (public)
- **Sorumluluk:**
  - Tüm dış trafiği karşılar; routing, JWT validation, request/response logging.
  - Token Bucket rate limit (Redis backed).
  - SSE/WebSocket bağlantılarını backend'e proxy.
  - CORS, TLS termination, header propagation (`X-Request-Id`, `X-User-Id`).
- **Patterns:** Edge Gateway, Auth Filter Chain, Rate Limit (Redis Lua), Circuit Breaker (Resilience4j).
- **Bağımlılıklar:** Keycloak (JWKS endpoint), Redis.
- **Servisler dışa açılmaz** → her şey gateway'den geçer.

#### 2.2 `identity-service` — Kullanıcı & Yetki
- **Stack:** Spring Boot 3.5 + Spring Security + Java 21
- **Port:** 8081 (internal)
- **DB:** PostgreSQL (`identity_db`) — users, profiles, custom_instructions
- **IAM Backend:** Keycloak (token issuance burada) — bu servis Keycloak'ı yönetir + ek profil verilerini tutar.
- **Endpoints:**
  - `POST /api/v1/auth/register` — kayıt (Keycloak'a delege)
  - `POST /api/v1/auth/login` — login (Keycloak token endpoint proxy)
  - `POST /api/v1/auth/refresh`
  - `POST /api/v1/auth/logout`
  - `GET  /api/v1/users/me`
  - `PUT  /api/v1/users/me/profile` — display name, avatar, language, theme
  - `GET  /api/v1/users/me/custom-instructions`
  - `PUT  /api/v1/users/me/custom-instructions` — "you are talking to..." / "respond like..."
- **Eventler (publish):**
  - `user.registered.v1` → consumer: billing (free plan oluştur), notification (welcome email)
  - `user.profile-updated.v1`
  - `user.deleted.v1` → cascade conversation/asset cleanup
- **Patterns:** Transactional Outbox (events), Idempotency (register endpoint).

#### 2.3 `conversation-service` — Sohbet & Mesaj
- **Stack:** Spring Boot 3.5 + Spring Data JPA + Java 21
- **Port:** 8082 (internal)
- **DB:** PostgreSQL (`conversation_db`) — chats, messages, message_outbox, idempotency_keys
- **Search:** Elasticsearch (CQRS read model — `chat-search` index, async populated via Kafka).
- **Endpoints:**
  - `POST   /api/v1/chats`                       — yeni sohbet (boş)
  - `GET    /api/v1/chats?page=&size=`           — listele (kullanıcının)
  - `GET    /api/v1/chats/{chatId}`              — detay + mesajlar
  - `PUT    /api/v1/chats/{chatId}`              — başlık değiştir
  - `DELETE /api/v1/chats/{chatId}`
  - `GET    /api/v1/chats/search?q=`             — Elasticsearch'e gider
  - `POST   /api/v1/chats/{chatId}/messages`     — kullanıcı mesajı persist (Idempotency-Key zorunlu)
  - `GET    /api/v1/chats/{chatId}/messages?cursor=` — pagination
- **Eventler (publish):**
  - `message.user-sent.v1` → inference orchestrator dinler → akış başlatır
  - `message.assistant-completed.v1` → telemetry, billing dinler
  - `chat.created.v1`, `chat.deleted.v1`
- **Eventler (subscribe):**
  - `inference.completed.v1` → assistant message'ı persist et (Saga step)
- **Patterns:**
  - **Transactional Outbox** (DB ve Kafka tutarlılığı için): mesaj insert + outbox row tek transaction'da, polling job Kafka'ya basar.
  - **Idempotency** (`Idempotency-Key` header) — duplicate POST'ları engeller.
  - **CQRS** — yazma PostgreSQL, okuma için search Elasticsearch.

#### 2.4 `inference-orchestrator` — Saga Orchestrator + SSE Stream
- **Stack:** Spring Boot WebFlux 3.5 + Java 21 (reactive)
- **Port:** 8083 (internal — gateway streaming pass-through)
- **DB:** PostgreSQL (`inference_db`) — saga_state, inference_jobs (in-flight tracking için)
- **Sorumluluk:**
  - Kullanıcı mesajı geldiğinde **Saga** akışını yönetir:
    1. `guardrail-service` → prompt güvenli mi?
    2. `router-service` → hangi modele gitsin?
    3. `billing-service` → kullanıcının kotası var mı?
    4. `inference-worker-*` → token streaming başlar (SSE/gRPC stream).
    5. Token'lar gateway'e relay edilir → frontend'e SSE.
    6. Tamamlanınca → `inference.completed.v1` event publish.
    7. Hata olursa **compensating transactions** (kotayı geri ver, jobu fail et).
- **Endpoints:**
  - `POST /api/v1/inference/stream` (SSE) — body: `{chatId, message, modelHint?, files?}`
- **Patterns:**
  - **Saga (Orchestration)** — merkezi koordinasyon (choreography'den daha izlenebilir).
  - **Circuit Breaker** her downstream call için (Resilience4j).
  - **Bulkhead** — model çağrılarını ayrı thread pool'larda izole et.
  - **Timeout + Retry** (idempotent stepler için).
  - **Backpressure** (WebFlux Flux ile) — yavaş client'lar memory yemiyor.

#### 2.5 `inference-worker-goktug` — GoktugGPT Inference Engine
- **Stack:** Python 3.11 + FastAPI + PyTorch + Uvicorn (ASGI)
- **Port:** 9001 (internal)
- **GPU:** Required for production (CPU fallback for dev)
- **Sorumluluk:**
  - `goktugGPT/src/model/` altındaki modeli yükler (`best_model.pt` + `tokenizer.json`).
  - SSE veya gRPC stream üzerinden token-by-token üretir.
  - **Continuous batching** + **KV cache** — birden fazla istemi aynı GPU'da paketler.
  - **Greedy / top-k / top-p / temperature** sampling parametreleri.
- **Endpoints:**
  - `POST /v1/generate` (SSE stream) — body: `{prompt, params, jobId}`
  - `GET  /v1/health` — model yüklü mü, GPU memory durumu
  - `GET  /v1/models` — yüklü modellerin listesi
- **Patterns:** Worker pool, async queue (asyncio), graceful shutdown, prometheus metrics.

#### 2.6 `asset-service` — Dosya & Blob Yönetimi
- **Stack:** Spring Boot 3.5 + Java 21
- **Port:** 8084 (internal)
- **DB:** PostgreSQL (`asset_db`) — assets metadata
- **Blob:** MinIO (S3-compatible) — `gpt-assets` bucket
- **Endpoints:**
  - `POST /api/v1/assets/upload-url` — presigned PUT URL üret (frontend direkt MinIO'ya basar; backend'in I/O'sunu yormaz)
  - `POST /api/v1/assets/confirm` — upload tamam, validate (Tika ile MIME), DB'ye yaz
  - `GET  /api/v1/assets/{assetId}/download-url` — presigned GET URL
  - `DELETE /api/v1/assets/{assetId}`
- **Eventler:**
  - `asset.uploaded.v1` → multimodal-pipeline (Faz 2) dinleyip OCR/STT yapar
  - `asset.deleted.v1`
- **Patterns:** Presigned URL (backend bypass), Idempotency, Background validation.

> **Faz 1 sonu hedefi:** Kullanıcı kayıt olur → giriş yapar → sohbet açar → yazı yazar → goktugGPT modelinden streaming cevap alır → geçmişi görür → file upload edebilir.

---

### Faz 2 — Özellikler & Gerçekçi Üretim

#### 2.7 `router-service` — Semantic Model Router
- **Stack:** Python 3.11 + FastAPI
- **Port:** 9002
- **Sorumluluk:** Prompt'un zorluğunu/içeriğini analiz eder; küçük embedding modeliyle `{simple-chat, code, math, reasoning}` sınıflandırır. Çıktı: hangi model kullanılsın (goktug-mini / goktug-pro / external-claude vs).
- **Patterns:** ML-as-a-Service, in-process embedding cache.

#### 2.8 `guardrail-service` — Prompt Injection & Toxicity
- **Stack:** Python 3.11 + FastAPI + transformers (DeBERTa, Detoxify)
- **Port:** 9003
- **Endpoints:** `POST /v1/check` — body: `{text}` → `{safe: bool, categories: [...], score}`
- **Patterns:** Sync REST (low latency hot path), in-memory model.

#### 2.9 `billing-service` — Quota & Subscription
- **Stack:** Spring Boot 3.5
- **Port:** 8085
- **DB:** PostgreSQL (`billing_db`) — subscriptions, usage_records, plans
- **Cache:** Redis (token bucket per user)
- **Endpoints:**
  - `GET  /api/v1/billing/me/quota` — kalan token / istek sayısı
  - `GET  /api/v1/billing/me/subscription`
  - `POST /api/v1/billing/me/subscribe` — Stripe webhook entegrasyonu (Faz 3)
- **Eventler (subscribe):**
  - `inference.completed.v1` → token kullanımı düş
  - `user.registered.v1` → free plan oluştur
- **Patterns:** Token Bucket (Redis Lua), CQRS, Event Sourcing (usage_records audit trail).

#### 2.10 `telemetry-consumer` — Logs & RLHF Pipeline
- **Stack:** Spring Boot 3.5 + Spring Kafka
- **Port:** 8086 (internal — yalnız Kafka consumer)
- **Sink:** Elasticsearch (analytics) + S3/Azure DLS (cold storage for RLHF training data)
- **Subscribe:**
  - `inference.completed.v1`, `message.user-sent.v1`, `feedback.given.v1`
- **Patterns:** Stream Processing, Batching to data lake, Schema evolution.

#### 2.11 `notification-service` — Bildirimler
- **Stack:** Spring Boot 3.5 + WebSocket (STOMP)
- **Port:** 8087
- **Sorumluluk:** WebSocket connection manager, async job tamamlandı bildirimi, e-posta gönder.
- **Subscribe:** `notification.requested.v1`, `user.registered.v1` (welcome email)

---

### Faz 3 — Kurumsal & İleri Seviye

#### 2.12 `rag-service` — Knowledge Retrieval
- **Stack:** Python 3.11 + FastAPI + LangChain + pgvector (veya Milvus)
- **Port:** 9004
- **Sorumluluk:** Doküman embedding, hibrit retrieval (BM25 + dense vector), context injection.

#### 2.13 `tool-service` — MCP & Tool Orchestration
- **Stack:** Node.js 20 + TypeScript + `@langchain/mcp-adapters`
- **Port:** 7001
- **Sorumluluk:** Web search (Google CSE), code sandbox (Docker-in-Docker), function calling bridge.

#### 2.14 `audit-service` — Compliance & WORM Logging
- **Stack:** Spring Boot 3.5
- **DB:** PostgreSQL append-only + S3 Object Lock (WORM).
- Subscribe: tüm `*.v1` eventler.

#### 2.15 `feature-flag-service` — A/B & Rollout
- **Stack:** Spring Boot 3.5 + Unleash (open source) entegrasyonu.

---

## 3. Mimari Patternler — Implementation Notes

### 3.1 Saga Pattern (Orchestration)

**Use case:** Kullanıcı mesajı gönderir → guardrail check → quota check → inference → persist → billing update.

**Implementation:** `inference-orchestrator` servisi merkezi orchestrator. Her step için:
- Sync REST/gRPC call (timeout'lu).
- Başarısızlıkta **compensating action**.
- State `saga_state` tablosunda persist edilir → orchestrator restart olsa bile kaldığı yerden devam eder.

**State machine:**
```
PENDING → GUARDRAIL_OK → QUOTA_OK → INFERENCE_STREAMING → COMPLETED
   ↓           ↓             ↓              ↓
 FAILED    BLOCKED       QUOTA_EXCEEDED  INFERENCE_FAILED
                                              ↓
                                         (compensate: refund quota)
```

### 3.2 Transactional Outbox

**Use case:** `conversation-service` mesajı DB'ye yazıyor + Kafka'ya event basıyor. İkisinin atomic olması lazım.

**Implementation:**
1. Mesaj insert + `outbox` tablosuna event row tek transaction.
2. Ayrı bir scheduler / Debezium CDC `outbox` tablosunu okur, Kafka'ya basar, row'u `processed=true` işaretler.
3. **At-least-once delivery** garantili → consumer tarafta idempotency.

### 3.3 Idempotency

**Use case:** Frontend retry yaparsa duplicate mesaj yaratmasın.

**Implementation:** Her POST endpoint `Idempotency-Key` header zorunlu. Servis bu key + userId + endpoint'i `idempotency_keys` tablosunda 24h tutar; aynı key geldiyse cached response döner.

### 3.4 CQRS

**Use case:** Chat title search → PostgreSQL `LIKE %x%` çok yavaş, milyonlarca chat'te N+1 olur.

**Implementation:**
- **Write:** PostgreSQL (`chats` tablosu).
- **Read:** Elasticsearch `chat-search` index. `chat.created.v1` / `chat.title-changed.v1` eventlerini consume eden async indexer (`conversation-service` içinde projection worker veya ayrı search-projector servisi).

### 3.5 Circuit Breaker / Bulkhead / Retry

**Library:** Resilience4j (Spring Boot ile native entegrasyon).

```java
@CircuitBreaker(name = "inference-worker", fallbackMethod = "inferenceFallback")
@Retry(name = "inference-worker")
@Bulkhead(name = "inference-worker", type = Bulkhead.Type.THREADPOOL)
@TimeLimiter(name = "inference-worker")
public CompletableFuture<...> callInferenceWorker(...) { ... }
```

Konfigürasyonlar `application.yml`'de:
- `inference-worker`: 50% hata oranı, 10s window → açık, 30s sonra half-open
- Retry: 3 deneme, exponential backoff (200ms, 400ms, 800ms)
- Timeout: 60s (streaming için uzun)
- Bulkhead: max 100 concurrent inference call

### 3.6 Event Schema

Tüm eventler **JSON Schema** ile versiyonlanır (`shared-contracts/events/`). İsim formatı: `<domain>.<event>.v<version>` örn: `message.user-sent.v1`.

Her event şunları içerir:
```json
{
  "eventId": "uuid",
  "eventType": "message.user-sent.v1",
  "occurredAt": "2026-05-04T12:34:56Z",
  "producer": "conversation-service",
  "traceId": "...",
  "userId": "...",
  "payload": { ... }
}
```

---

## 4. Teknoloji Seçimleri

| Katman | Seçim | Alternatif | Neden |
|--------|-------|-----------|-------|
| Edge Gateway | Spring Cloud Gateway | Kong, Envoy | Java ekosistemiyle uyum, reactive |
| IAM | Keycloak | Auth0 (paid), home-grown | OSS, OAuth2/OIDC tam destek |
| iş servisleri | Spring Boot 3.5 + Java 21 | .NET, Go | Mevcut sarıtaygpt deneyimi |
| AI servisleri | Python 3.11 + FastAPI | Flask, Triton | Pytorch native, ML ekosistemi |
| MCP/Tool | Node.js + TypeScript | Python | langchain-mcp-adapters native |
| Async | Apache Kafka | RabbitMQ | High throughput, replay, partitioning |
| Sync (iç) | gRPC + REST | Sadece REST | Streaming, schema |
| RDBMS | PostgreSQL 16 | MySQL | JSON/jsonb, pgvector, mature |
| Search | Elasticsearch 8 | OpenSearch | Mature, Spring Data ES |
| Cache | Redis 7 | Memcached | Lua scripts, pubsub, persistence |
| Vector DB | pgvector → Milvus | Pinecone (paid) | Önce simple, gerekirse scale up |
| Blob | MinIO (dev) → S3 (prod) | Azure Blob | S3 API standart |
| Container | Docker + docker-compose (dev) | — | — |
| Orchestration | Kubernetes + Helm (prod) | Nomad | Endüstri standartı |
| Service Mesh | Istio (Faz 3) | Linkerd | mTLS, traffic split |
| Observability | OpenTelemetry → Jaeger + Prometheus + Grafana + ELK | Datadog (paid) | OSS stack |
| CI/CD | GitHub Actions + ArgoCD | GitLab CI | Repo zaten GitHub'da |
| Secrets | HashiCorp Vault (Faz 3) | K8s Secrets | Rotation, audit |

---

## 5. Repo Yapısı

```
goktugGPT/
├── goktugGPT/                    # MEVCUT — model, tokenizer, eğitim kodu (Python)
│   ├── src/
│   ├── train.py
│   └── ...
│
└── platform/                     # YENİ — mikroservis monorepo
    ├── MICROSERVICES_PLAN.md     # bu döküman
    ├── PHASE2_TODO.md            # faz 2 detayları
    ├── PHASE3_TODO.md            # faz 3 detayları
    ├── docker-compose.yml        # tüm infra + servisler (dev)
    ├── docker-compose.infra.yml  # sadece altyapı
    │
    ├── infra/                    # IaC, K8s manifests, helm charts
    │   ├── k8s/
    │   ├── helm/
    │   └── observability/
    │
    ├── shared-contracts/         # tüm servislerin paylaştığı şemalar
    │   ├── events/               # Kafka event JSON schemas
    │   │   ├── user.registered.v1.json
    │   │   ├── message.user-sent.v1.json
    │   │   └── ...
    │   ├── openapi/              # OpenAPI 3.0 specs
    │   │   ├── identity.yaml
    │   │   ├── conversation.yaml
    │   │   └── ...
    │   └── proto/                # gRPC .proto dosyaları (iç iletişim)
    │
    ├── libs/                     # ortak Java/Python kütüphaneleri
    │   ├── common-java/          # ortak DTO, exception, JWT util
    │   └── common-python/        # ortak telemetry, pydantic models
    │
    └── services/
        ├── api-gateway/          # Java
        ├── identity-service/     # Java
        ├── conversation-service/ # Java
        ├── inference-orchestrator/  # Java (WebFlux)
        ├── inference-worker-goktug/ # Python (FastAPI)
        ├── asset-service/        # Java
        ├── billing-service/      # Java         [Faz 2]
        ├── guardrail-service/    # Python       [Faz 2]
        ├── router-service/       # Python       [Faz 2]
        ├── telemetry-consumer/   # Java         [Faz 2]
        ├── notification-service/ # Java         [Faz 2]
        ├── rag-service/          # Python       [Faz 3]
        ├── tool-service/         # Node         [Faz 3]
        ├── audit-service/        # Java         [Faz 3]
        └── feature-flag-service/ # Java         [Faz 3]
```

---

## 6. İlerleme Durumu (Bir Sonraki Session İçin)

> **Bu bölümü her session sonunda güncelle.**

### Bu session'da tamamlananlar
- [x] Master plan dökümanı (`MICROSERVICES_PLAN.md`) yazıldı
- [x] `platform/` dizin iskeleti kuruldu
- [x] `infra/docker-compose.yml` (Postgres ×5, Redis, Kafka, Keycloak, MinIO, Jaeger, Prometheus, Grafana, Kibana)
- [x] Maven parent POM
- [x] Event schema'lar (`message.user-sent.v1`, `inference.completed.v1`, `user.registered.v1`, `asset.uploaded.v1`, `envelope.v1`)
- [x] Keycloak realm export (`goktuggpt` realm, `goktuggpt-web` PKCE client, `goktuggpt-internal`)
- [x] Faz 1 servisleri için iskelet **+ TAM IMPLEMENTASYON**:
  - **api-gateway** — Spring Cloud Gateway + Security + RateLimit
  - **identity-service** — Profile/CustomInstructions entity'leri + KeycloakClient (admin API + token proxy) + AuthController + UserController + Outbox pattern (`user.registered.v1`)
  - **conversation-service** — Chat/Message entity + JPA repos + ChatService + MessageService + ChatController + MessageController + IdempotencyService (cache-aside) + OutboxPublisher + OutboxPoller + InferenceCompletedConsumer (Saga step) + GlobalExceptionHandler
  - **inference-orchestrator** — InferenceJob entity + WebClientConfig + GuardrailClient + BillingClient + InferenceWorkerClient (SSE) + InferenceSaga (full reactive chain with Resilience4j CB) + InferenceEventPublisher + InferenceController (POST /stream)
  - **asset-service** — AssetEntity + AssetRepository + MinioConfig (S3Client + Presigner) + PresignedUrlService + AssetService (upload-url → confirm with HEAD validation → publish asset.uploaded.v1) + AssetController
  - **inference-worker-goktug** — FastAPI app + ModelHandle + StreamingGenerator (asyncio.to_thread for non-blocking forward pass) + Prometheus metrics + SSE generate endpoint
- [x] Faz 2 (`PHASE2_TODO.md`) ve Faz 3 (`PHASE3_TODO.md`) detaylı yol haritaları

### Bir sonraki session'da yapılacaklar (öncelikli)

#### Smoke test ortamını ayağa kaldırma
1. `mvn -pl services/identity-service -am package` — derleme hatalarını giderme (Lombok plugin Maven path'inde olmalı, hibernate-types'ın doğru sürümü)
2. `docker compose --profile infra up -d` — altyapı kontrolü
3. Keycloak realm import doğrulama: http://localhost:8180 → goktuggpt realm var mı?
4. Postgres'lerin healthy gelmesi (`docker compose ps`)
5. `docker compose up -d --build identity-service conversation-service inference-orchestrator inference-worker-goktug asset-service api-gateway`

#### Eksik kalan integration parçaları (✅ = bu session tamamlandı)
6. ✅ **api-gateway → JWT claims → header mapper filter:** `JwtToHeadersFilter.java` (GlobalFilter) — `sub`→X-User-Id, `email`→X-User-Email, `realm_access.roles`→X-User-Roles. Anonymous endpoint'lerde header strip ediliyor (spoofing engellendi). + `CorsConfig` eklendi.
7. ✅ **conversation-service → Elasticsearch projector:** `ChatSearchProjector` (Kafka consumer for `chat.events`) + `ChatSearchDocument` + `ChatSearchRepository` + `ChatSearchController` (`GET /api/v1/chats/search`).
8. ✅ **inference-orchestrator recovery worker:** `SagaRecoveryWorker` — her 30s'de stale job'ları (>10dk update yok, terminal değil) `INFERENCE_FAILED`'a çeker + refund tetikler. `findStaleJobs` repository query.
9. **inference-orchestrator → conversation-service**: kullanıcı mesajı zaten `message.user-sent.v1` event'i ile yayınlandı; orchestrator bu event'i consume edip Saga başlatabilir (Faz 2 — şu an REST trigger). [hâlâ todo]
10. **inference-worker-goktug**: best_model.pt + tokenizer.json'ın `goktugGPT/checkpoints/` altında olması lazım. Kaggle eğitimi tamamlanana kadar dummy fallback ekle. [hâlâ todo]

#### Kalite / Operasyon
11. Resilience4j konfigleri tüm Spring servislere ekle (yalnız orchestrator'da var) [todo]
12. ✅ **OpenTelemetry tüm Java servislerinde:** `opentelemetry-javaagent.jar` v2.7.0 her Dockerfile'a `-javaagent` flag'iyle eklendi + docker-compose her servise `OTEL_*` env vars (SERVICE_NAME, RESOURCE_ATTRIBUTES, EXPORTER_OTLP_ENDPOINT). Auto instrumentation: HTTP, JDBC, Kafka producer/consumer, WebClient, Spring Webflux. Detay: [`infra/observability/OBSERVABILITY.md`](infra/observability/OBSERVABILITY.md).
13. Testcontainers ile integration test'ler (OutboxPoller flow, Saga happy path) [todo]
14. CI: GitHub Actions per-service build + test + Docker image push (GHCR) [todo]
15. Frontend planı (`FRONTEND_PLAN.md`) — kullanıcı tasarım kararını verince başlanır [todo]
16. ✅ **E2E smoke test:** [`docs/SMOKE_TEST.md`](docs/SMOKE_TEST.md) detaylı senaryo + [`scripts/smoke-test.sh`](scripts/smoke-test.sh) (bash) + [`scripts/smoke-test.ps1`](scripts/smoke-test.ps1) (PowerShell) — health check → register → chat → message (idempotency replay) → SSE inference → assistant persist verification → ES search verification.

### Faz 2'ye geçmeden önce karar verilecekler
- Keycloak yerine kendi auth-service'i mi? (decision: Keycloak — karar verildi)
- Vector DB pgvector mı Milvus mı? (decision: pgvector başlangıç, lazım olursa Milvus)
- Streaming SSE mi gRPC mi? (decision: dış SSE, iç gRPC)
- Frontend'in protokolu? (TBD — frontend planı ayrı yapılacak)

### Bilinen riskler / teknik borç
- `goktugGPT` modeli henüz yeterince kaliteli cevap üretmiyor (önceki konuşma) — `prepare_data.py` + Kaggle eğitimi tamamlanmadan inference-worker'ın gerçek değeri test edilemez.
- GPU yokken inference-worker CPU'da çok yavaş — dev environment'ta küçük model kullan.
- Kafka'sız local dev için `docker-compose up postgres redis` yeterli mi? — başlangıçta evet, event'leri Faz 2'de aç.

---

## 7. Mimar Notu — DON'Ts

**Bu listeyi her PR review'da kontrol et.**

- ❌ İki servis aynı DB'ye dokunmasın. (Database-per-Service kuralı.)
- ❌ Servis A → Servis B → Servis C zincirleri oluşturma. Sync chain >2 derinse → event-driven yap.
- ❌ Eventleri "message" gibi belirsiz isimlerle yayma. `<domain>.<event>.<version>` formatına uy.
- ❌ Versiyonsuz event ekleme. `v1`, `v2` zorunlu.
- ❌ `@Transactional` ile cross-service işlem yapmaya çalışma. Saga kullan.
- ❌ Servisler arası DTO paylaşımını "common-jar"a koyma — koupling yaratır. Her servis kendi DTO'sunu üretir; sadece **event schema** paylaşılır.
- ❌ Frontend doğrudan iç servise bağlanmasın. Her şey gateway'den geçer.
- ❌ Secret'ları `application.yml`'de hardcode etme. `${ENV_VAR}` veya Vault.
- ❌ Logging'i `System.out.println` ile yapma. SLF4J + structured JSON + traceId.

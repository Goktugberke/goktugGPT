# goktugGPT — Backend

The **event-driven microservices platform** that turns the custom transformer LLM
([`../ai-model/`](../ai-model/)) into a real, multi-user SaaS. ~12 application
services (Java + Python) plus their infrastructure, all orchestrated with Docker
Compose. For the project overview and tech-stack glossary see the
[root README](../README.md).

---

## Design principles

- **Database-per-service** — every service owns its own PostgreSQL DB. Services never read each other's tables; they communicate only via HTTP (through the gateway) or **Kafka events**.
- **Event-driven** — state changes are published as `*.v1` events. Producers don't know who consumes them (loose coupling).
- **Layered (3-tier) services** — `api → service → repository`, not hexagonal; kept deliberately simple.
- **Stateless app services behind the gateway** — the gateway validates JWTs once and forwards the user identity as headers, so downstream services don't re-validate.

---

## Services

| Service | Port | Lang | Role |
|---|---|---|---|
| **api-gateway** | 8080 | Java (WebFlux) | Single entry point. Validates Keycloak JWT, maps it to `X-User-*` headers, routes to services. |
| **identity-service** | 8081 | Java | Register/login. Manages users in Keycloak, stores profiles, emits `user.registered.v1`. |
| **conversation-service** | 8082 | Java | Chats & messages. Emits `message.user-sent.v1`; CQRS search via Elasticsearch. |
| **inference-orchestrator** | 8083 | Java (WebFlux) | The brain. Runs the Saga (guardrail → router → billing → model), **streams** tokens via SSE, emits `inference.completed.v1`. |
| **asset-service** | 8084 | Java | File uploads via MinIO **presigned URLs**; emits `asset.uploaded.v1`. |
| **billing-service** | 8080 | Java | Token quota & usage. Redis token-bucket rate limiting; consumes registration + inference events. |
| **telemetry-consumer** | 8080 | Java | Consumes all `*.v1` events → Elasticsearch (analytics) + MinIO (cold storage). |
| **notification-service** | 8080 | Java | WebSocket push + email; reacts to registration/inference events. |
| **config-server** | 8888 | Java | Spring Cloud Config Server — serves shared config from a Git repo. |
| **llm-server** | 9001 | Python | Serves the trained transformer (loads `best_model.pt`). Token streaming. |
| **guardrail-service** | 9003 | Python | Prompt-injection / toxicity / PII checks (Saga's first step). |
| **router-service** | 9002 | Python | Classifies the prompt → picks which model to use. |

> `llm-server` is the bridge between the two worlds: **`ai-model/` trains → produces
> `best_model.pt` → `llm-server` serves it.** It lives here because it's a deployed
> microservice (in compose, called by the orchestrator).

---

## Infrastructure

| Component | Port | What it does |
|---|---|---|
| **PostgreSQL** ×6 | 5432 (internal) | One DB per stateful service (identity, conversation, inference, asset, billing, keycloak). |
| **Apache Kafka** (+ Zookeeper) | 9092 | Event bus — async, ordered, durable messaging between services. |
| **Keycloak** | 8180 | Identity provider (OAuth2/OIDC), JWT issuer, role management. |
| **Redis** | 6379 | In-memory store for billing rate-limit token buckets. |
| **MinIO** | 9000 / 9090 | S3-compatible object storage for user assets + telemetry cold storage. |
| **Elasticsearch** + **Kibana** | 9200 / 5601 | Chat search (CQRS) + telemetry analytics + its visual UI. |
| **Apicurio Schema Registry** | 8086 | Versioned JSON Schema event contracts + backward-compat enforcement. |
| **Jaeger** | 16686 | Distributed tracing UI (OpenTelemetry collector on 4317/4318). |
| **Prometheus** | 9099 | Scrapes service metrics. |
| **Grafana** | 3000 | Dashboards over Prometheus metrics. |

*(See the [root README tech glossary](../README.md#tech-stack--what-each-piece-is--why) for a one-line "what is X" on each of these.)*

---

## Architecture patterns

- **Saga (orchestration)** — `inference-orchestrator` coordinates the multi-service inference flow and handles partial failure. SSE stream finalized with `doFinally` so the saga completes even if the client disconnects.
- **Transactional Outbox** — services (e.g. conversation, identity) write domain row + event in one DB transaction; a poller publishes to Kafka. Guarantees no lost events.
- **Idempotency-Key** — `conversation-service` dedupes repeated message sends by key, returning the cached response.
- **CQRS** — conversation writes to Postgres; a projector mirrors chats into Elasticsearch for `GET /api/v1/chats/search?q=`.
- **SSE token streaming** — orchestrator → gateway → browser, token by token.

---

## Event catalog

Events flow on Kafka as JSON envelopes (`eventId`, `eventType`, `occurredAt`, `producer`, `payload`). Schemas are versioned in [`schemas/events/`](schemas/events/) and registered to Apicurio.

| Event | Producer | Consumers |
|---|---|---|
| `user.registered.v1` | identity | billing (free plan), notification (welcome) |
| `message.user-sent.v1` | conversation | inference-orchestrator (starts saga) |
| `chat.created/title-changed/deleted.v1` | conversation | search projector (Elasticsearch) |
| `inference.completed.v1` | orchestrator | conversation (persist answer), billing (usage), notification |
| `inference.failed.v1` | orchestrator | notification |
| `asset.uploaded.v1` | asset | telemetry |

Register all schemas (registry must be up):
```bash
./scripts/register-schemas.ps1     # or .sh
```

---

## Directory layout

```
backend/
├── docker-compose.yml          # all ~29 containers
├── pom.xml                     # parent Maven reactor (multi-module)
├── .dockerignore
├── start.bat / stop.bat        # one-click up/down (Windows)
│
├── api-gateway/ identity-service/ conversation-service/
├── inference-orchestrator/ asset-service/ billing-service/
├── telemetry-consumer/ notification-service/ config-server/
├── guardrail-service/ router-service/ llm-server/   # (llm-server = git submodule)
│
├── config-repo/                # Spring Cloud Config Git backend (runtime, git-ignored)
├── config-server/seed-config/  # version-controlled seed for config-repo
├── schemas/events/             # JSON Schema event contracts
├── shared-contracts/           # OpenAPI + event contracts
├── infra/                      # observability configs, keycloak realm, Helm charts
│   └── helm/goktug-platform/   # umbrella Helm chart (dev/prod values)
└── scripts/                    # smoke-test, schema registration, config-repo init
```

---

## Running

```bash
cd backend
./start.bat               # launches Docker, `docker compose --profile all up -d`, waits for health
./scripts/smoke-test.ps1  # end-to-end check (register → chat → stream → asset): expect 6/6
./stop.bat
```

**Config server note:** `config-repo/` is a Git repo read by Spring Cloud Config and
is git-ignored. On a fresh clone it's recreated from `config-server/seed-config/` by
`scripts/init-config-repo.ps1` (which `start.bat` runs automatically).

**Tracing:** services export traces to Jaeger (OTLP gRPC). Open http://localhost:16686,
pick `inference-orchestrator`, and a single chat request appears as one trace spanning
gateway → conversation → orchestrator → billing → (Kafka) → notification/telemetry.

---

## Kubernetes (Helm)

The umbrella chart in [`infra/helm/goktug-platform/`](infra/helm/goktug-platform/)
deploys all app services (one generic template ranging over a `services` map), with
`values-dev.yaml` / `values-prod.yaml` (replicas, HPA, resource limits, probes,
SSE-friendly ingress). Infra (Postgres, Kafka, etc.) is expected to be provided
in-cluster. See the chart's own README.

---

## Status

Phase 1 (MVP) ✅ · Phase 2 hardening ✅ (tracing, config, schema registry, CQRS, Helm) ·
Testcontainers integration tests ⏳ deferred. See [`../docs/`](../docs/) for full phase checklists.

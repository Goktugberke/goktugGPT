# goktugGPT

A self-hosted, **ChatGPT/Claude-style AI chat platform** built around a **custom
transformer LLM written entirely from scratch** (no OpenAI, no Hugging Face, no
pretrained weights) and a **production-grade microservices backend**.

The project has three parts:

| Area | Path | What it is | Status |
|---|---|---|---|
| 🧠 **AI Model** | [`ai-model/`](ai-model/) | The from-scratch PyTorch transformer — BPE tokenizer, multi-head attention, decoder blocks, `<think>` reasoning, training loop. | ✅ Working |
| ⚙️ **Backend** | [`backend/`](backend/) | Event-driven microservices platform that turns the model into a real SaaS: auth, chats, streaming inference, billing, assets, observability. | ✅ ~MVP + hardening done |
| 🎨 **Frontend** | [`frontend/`](frontend/) | Web UI (React). | ⏳ **To be added** |

> Detailed docs live next to the code: **[ai-model/README](ai-model/README.md)**
> (how the model works + training) and **[backend/README](backend/README.md)**
> (services, patterns, how to run).

---

## What is goktugGPT?

goktugGPT started as a **language model built from scratch** to demonstrate how a
modern LLM actually works at every layer — tokenization → embeddings → attention →
transformer → generation → reasoning. It then grew into a **full platform** that
serves that model the way a real product would: user accounts, persistent
conversations, token-by-token streaming, usage/billing, file uploads, and full
observability — all as independent, event-driven microservices.

In short: **a custom LLM + the entire backend infrastructure to run it as a service.**

---

## How it works (request flow)

A single chat message flows through the platform like this:

```
                         ┌──────────────┐
   Browser ──HTTPS──▶    │ API Gateway  │  validates JWT, routes, maps user → headers
                         └──────┬───────┘
                                ▼
                       ┌─────────────────┐   saves message, emits event
                       │ conversation-svc │ ───────────────┐
                       └─────────────────┘                 │ Kafka
                                                            ▼
                       ┌──────────────────────┐   Saga orchestration
                       │ inference-orchestrator│  guardrail → router → billing
                       └──────────┬────────────┘  → llm-server (stream)
                                  │ Server-Sent Events (token by token)
                                  ▼
                       custom transformer (ai-model, served by llm-server)
                                  │
   billing • telemetry • notifications  ◀── Kafka events (async fan-out)
```

- **Synchronous** path (browser → gateway → service) uses HTTP.
- **Asynchronous** path (between services) uses **Kafka** events — services never call each other's databases; they react to events.
- The model's answer is **streamed** back to the browser token by token (SSE), like the typing effect in ChatGPT.

---

## What has been built

**Phase 1 — MVP (done):** register → login → create chat → send message →
streaming model response → file upload, working end-to-end (smoke test 6/6).

**Phase 2 — Production hardening (mostly done):**
- ✅ Guardrail, router, billing, telemetry, notification services
- ✅ **Distributed tracing** (OpenTelemetry → Jaeger) — one chat request shows up as a single trace across 6+ services
- ✅ **Centralized config** (Spring Cloud Config, Git-backed)
- ✅ **Schema Registry** (Apicurio) — versioned event contracts with backward-compatibility enforcement
- ✅ **CQRS** chat search (Elasticsearch read model)
- ✅ **Helm charts** for Kubernetes deployment
- ✅ Self-do guides for **Istio** service mesh and **GitHub Actions CI/CD**
- ⏳ Testcontainers integration tests (deferred)

**Frontend & Phase 3** (RAG, tool use, multimodal, fine-tuning, etc.) — future work.

See [`docs/`](docs/) for the full microservices plan and phase checklists.

---

## Tech Stack — what each piece is & why

> A quick "what is this?" for every major technology, so the stack is legible at a glance.

### Languages & frameworks
- **Java 21 + Spring Boot** — the backbone of all backend microservices. Spring Boot handles the web server, DB access, and wiring so the code focuses on business logic.
- **Python + PyTorch** — the AI model (`ai-model/`) and the ML-adjacent services (model serving, guardrail, router). PyTorch is the neural-network framework (tensors, autograd, GPU).
- **Maven** — Java build tool; one parent POM manages versions for all services (multi-module).

### Data stores
- **PostgreSQL** — relational database. Each service has its **own** Postgres instance (database-per-service) so services stay decoupled.
- **Redis** — in-memory key-value store; used for billing rate-limiting (token buckets) and fast counters. Sub-millisecond because it's RAM, not disk.
- **Elasticsearch + Kibana** — search/analytics engine. Powers chat full-text search (CQRS read model) and stores telemetry. **Kibana** is its visual UI for querying/dashboards.
- **MinIO** — S3-compatible object storage (file/blob store) for user uploads. Self-hosted drop-in for AWS S3; same API, easy cloud migration later.

### Messaging & integration
- **Apache Kafka (+ Zookeeper)** — the event bus. Services publish events; interested services subscribe. Kafka persists events to disk in order, so nothing is lost if a service is down. **Zookeeper** is Kafka's coordination helper.
- **Apicurio Schema Registry** — stores the JSON Schema of every event, versioned, and rejects backward-incompatible changes. Guarantees producers and consumers agree on event shape.

### Platform / cross-cutting
- **Keycloak** — identity & access management (IAM). Handles login, JWT token issuance, OAuth2/OIDC, roles — so we don't hand-roll auth. The gateway validates tokens against it.
- **Spring Cloud Gateway** — the single entry point (API gateway). Validates the JWT once, then forwards the user identity to internal services as headers; routes each path to the right service. Reactive, so it can carry long-lived SSE streams.
- **Spring Cloud Config** — central configuration. Common settings (Keycloak URL, tracing endpoint) are read from one Git-backed source instead of being duplicated per service.

### Observability (knowing what the system is doing)
- **OpenTelemetry + Jaeger** — **distributed tracing**. OpenTelemetry tags each step of a request; Jaeger stitches them into one timeline so you can see *where* a request slowed down or failed across many services.
- **Prometheus** — **metrics collection**. Each service exposes numbers (request rate, latency, error rate, memory); Prometheus scrapes and stores them as time series.
- **Grafana** — turns Prometheus metrics into **visual dashboards** and alerts. Prometheus stores the data; Grafana draws the graphs.

### Runtime & deployment
- **Docker + Docker Compose** — each service runs in an isolated container; Compose brings up all ~29 containers in the right order (healthchecks + dependencies) from one file.
- **Helm** — packages the services for **Kubernetes** (templated manifests + per-environment values). See [`backend/infra/helm/`](backend/infra/helm/).
- **Istio** (planned) — service mesh for automatic mTLS, traffic splitting/canary, sidecar observability. See the [setup guide](docs/guides/ISTIO_SETUP.md).
- **GitHub Actions** (planned) — CI/CD: build + test + push images on each change. See the [guide](docs/guides/GITHUB_ACTIONS_CICD.md). *(Jenkins is **not** used — GitHub Actions is the chosen CI.)*

---

## Architecture patterns

- **Saga (orchestration)** — a multi-service operation (guardrail → router → billing → model) coordinated from one place, with rollback on failure. The distributed-systems alternative to one big transaction.
- **Transactional Outbox** — write the DB row and the "to-be-sent" event in the *same* transaction; a poller publishes it to Kafka afterward. Prevents "saved to DB but event never sent" inconsistencies.
- **Idempotency-Key** — a repeated request (network retry) is detected by its key and returns the same result instead of creating duplicates.
- **CQRS** — writes go to Postgres; a separate read model in Elasticsearch powers search.
- **Server-Sent Events (SSE)** — streams the model's answer token by token to the browser (the ChatGPT typing effect).

---

## Running it (local)

Everything runs locally via Docker Compose. From the **`backend/`** directory:

```bash
cd backend
./start.bat          # Windows one-click: launches Docker, brings up all services, waits for health
# then verify:
./scripts/smoke-test.ps1
./stop.bat           # stop everything
```

First boot builds images and can take a few minutes. Key URLs once up:

| Service | URL |
|---|---|
| API Gateway | http://localhost:8080 |
| Keycloak (admin/admin) | http://localhost:8180 |
| Jaeger (tracing) | http://localhost:16686 |
| Schema Registry UI | http://localhost:8086/ui |
| Kafka UI | http://localhost:8090 |
| MinIO console | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| Kibana | http://localhost:5601 |

> Full backend details (every service, ports, patterns, troubleshooting): **[backend/README.md](backend/README.md)**.
> Training the model from scratch: **[ai-model/README.md](ai-model/README.md)**.

---

## Roadmap (high level)

- ✅ **Phase 1** — MVP platform (auth, chat, streaming inference, assets)
- ✅ **Phase 2** — production hardening (tracing, config, schema registry, CQRS, Helm) — *Testcontainers + load testing remain*
- ⏳ **Frontend** — React UI (in design)
- 🔭 **Phase 3** — RAG (chat with documents), tool/function calling, multimodal (PDF/image/audio), fine-tune orchestration, gRPC internal calls, Vault secrets, multi-region

Detailed plans: [`docs/MICROSERVICES_PLAN.md`](docs/MICROSERVICES_PLAN.md),
[`docs/PHASE2_TODO.md`](docs/PHASE2_TODO.md), [`docs/PHASE3_TODO.md`](docs/PHASE3_TODO.md).
AI-model roadmap (data scaling, real reasoning): [`ai-model/ROADMAP.md`](ai-model/ROADMAP.md).

---

*goktugGPT — a transformer LLM built from scratch, wrapped in a real microservices platform.*

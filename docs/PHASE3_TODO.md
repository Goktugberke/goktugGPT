# Phase 3 — Enterprise & State-of-the-Art

> Platform "production-ready" olduktan sonra rekabet edilebilir seviyeye taşımak için.

## Servisler

### 3.1 rag-service (Python + pgvector / Milvus)
- Doküman ingest (PDF/MD/HTML)
- Chunk + embed (`text-embedding-3-small` veya `bge-m3` lokal)
- Hibrit retrieval: BM25 + cosine similarity
- Inference-orchestrator saga'ya yeni step: `RAG_RETRIEVE` (opsiyonel — sadece "@knowledge" gibi flag varsa)

### 3.2 tool-service (Node.js + MCP)
- Web search (Google CSE, Brave API, SerpAPI)
- Code sandbox (Docker-in-Docker veya Firecracker microVM)
- Calculator, datetime, weather gibi built-in tools
- LLM "function calling" → tool dispatch

### 3.3 audit-service (Java)
- Tüm `*.v1` eventleri consume + WORM (Write Once Read Many) storage
- S3 Object Lock veya append-only PostgreSQL
- Compliance reports (GDPR data export, "kim hangi veriyi okudu")

### 3.4 feature-flag-service (Java + Unleash)
- Per-user / per-tenant feature toggle
- A/B test framework (router'da %5 traffic yeni modele)
- Hot reload (servis restart yok)

### 3.5 fine-tune-orchestrator (Python + Ray/Dask)
- Kullanıcı dataset upload → asset-service
- LoRA fine-tune job (background)
- Tamamlanınca yeni adapter'ı `inference-worker-goktug`'a hot deploy
- Notification → "fine-tuned model X is ready"

### 3.6 multimodal-pipeline (Python)
- `asset.uploaded.v1` consume → MIME'a göre dispatch:
  - PDF → Apache Tika text extraction
  - Image → BLIP-2 caption + CLIP embedding
  - Audio → Whisper STT
  - Video → frame extraction + STT
- Çıktı `assets` tablosuna `extracted_text`, `embeddings_id` olarak yazılır

### 3.7 abuse-detection-service (Python)
- Rate-based DDoS detection
- ML-based bot pattern (request inter-arrival time, prompt diversity)
- Output: `user.flagged.v1` event → identity-service ban'lar

### 3.8 cost-optimizer (Python)
- GPU utilization + queue depth metrics
- Dynamic routing: yüksek yük → free user'lar yavaş model'e
- Predictive scaling — saatlik load forecast

## Mimari Pattern'ler

### 3.9 Saga Pattern Maturity
- Saga'yı **Spring State Machine** library'sine refactor (custom enum yerine declarative state machine)
- Saga DSL ile yeni akışlar deklaratif tanımlanır

### 3.10 Event Sourcing (Tam — Faz 2 light idi)
- Conversation aggregate'i events'ten replay edilebilir
- "Time travel debugging": chat'in 2 saat önceki haline bak
- `chat-events` topic'i kalıcı (compaction kapalı)

### 3.11 CDC (Debezium)
- OutboxPoller yerine Debezium Postgres connector
- DB transaction commit edilir edilmez Kafka'ya WAL stream
- Daha düşük latency, daha az polling overhead

### 3.12 gRPC iç iletişim
- REST yerine internal servisler arası gRPC
- Protobuf schema (`.proto` dosyaları `shared-contracts/proto/`)
- Streaming: orchestrator → inference-worker SSE yerine gRPC bidi-stream

### 3.13 GraphQL Federation (frontend için)
- Her servis kendi GraphQL subgraph expose eder
- Apollo Federation gateway federated schema
- Frontend single endpoint'ten istediği veriyi çeker (over/under-fetching yok)

## Altyapı

### 3.14 Multi-region Deployment
- Aktif/aktif: 2 region, Kafka MirrorMaker
- Read replica DB her region'da
- Geo-routing (Cloudflare / Route53)

### 3.15 Vault for Secrets
- Tüm DB password, API key Vault'tan
- Auto rotation (90 gün)
- Service identity = Vault role (mTLS cert ile auth)

### 3.16 SLO/SLA Tracking
- Per-endpoint SLO (p95 < 500ms, error rate < 0.1%)
- Error budget tracking
- Alertmanager → PagerDuty/Slack on burn rate

### 3.17 Chaos Engineering
- Litmus / Chaos Mesh ile container kill, network partition
- Game days: "Kafka çökerse ne olur?"

### 3.18 Tenant Isolation (Enterprise plan için)
- Per-tenant DB schema veya ayrı DB (B2B müşterileri için)
- Per-tenant Kafka topic prefix
- Per-tenant model adapter'ı (3.5 fine-tune)

## Donanım / GPU

### 3.19 GPU Scheduler
- K8s'te NVIDIA device plugin
- Multi-instance GPU (MIG) — A100'ü 7 parçaya böl
- Inference worker'lar için VRAM-aware scheduling

### 3.20 Continuous Batching (vLLM-style)
- `inference-worker-goktug` içinde request queue
- N concurrent prompt'u tek forward pass'te batch
- Throughput 5-10x artar (GPU utilization > 90%)

### 3.21 KV Cache Management
- Multi-turn conversation'da prefix cache
- Aynı sistem prompt'u tekrar encode etmez
- Memory budget allocator (LRU eviction)

## DON'T (Faz 3'te bile yapma)

- ❌ Kendi vektör DB'ni yazma — pgvector / Milvus / Qdrant kullan
- ❌ Kendi LLM serving framework'ünü yazma — vLLM / TGI / Triton kullan
- ❌ Kendi feature flag sistemini yazma — Unleash / LaunchDarkly
- ❌ Kendi observability stack'ini topla — OpenTelemetry standardı + Jaeger/Tempo/Loki

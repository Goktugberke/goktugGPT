# goktugGPT — Documentation Index

ChatGPT/Claude-style AI SaaS platform built on the from-scratch goktugGPT model.

> This `docs/` folder holds the **planning & design docs**. For usage start with the
> top-level READMEs:
> - **[../README.md](../README.md)** — project overview, architecture, tech-stack glossary
> - **[../backend/README.md](../backend/README.md)** — services, patterns, how to run
> - **[../ai-model/README.md](../ai-model/README.md)** — the transformer model + training

## Documents here

| File | What |
|---|---|
| [MICROSERVICES_PLAN.md](MICROSERVICES_PLAN.md) | Master architecture plan |
| [PHASE2_TODO.md](PHASE2_TODO.md) | Phase 2 checklist (with current status) |
| [PHASE3_TODO.md](PHASE3_TODO.md) | Phase 3 (enterprise) backlog |
| [SMOKE_TEST.md](SMOKE_TEST.md) | End-to-end smoke test walkthrough |
| [guides/ISTIO_SETUP.md](guides/ISTIO_SETUP.md) | Istio service mesh — self-do guide |
| [guides/GITHUB_ACTIONS_CICD.md](guides/GITHUB_ACTIONS_CICD.md) | CI/CD — self-do guide |

## Quick start

Everything runs from the **`backend/`** directory:

```bash
cd backend

# Only infrastructure (Keycloak, Kafka, Postgres, MinIO, observability):
docker compose --profile infra up -d

# Everything (infra + all app services):
./start.bat                 # Windows one-click (or: docker compose --profile all up -d)

# One service locally with Maven (infra must be up):
mvn -pl conversation-service -am spring-boot:run
```

Infra UIs: Keycloak http://localhost:8180 · MinIO http://localhost:9090 ·
Kafka UI http://localhost:8090 · Jaeger http://localhost:16686 ·
Grafana http://localhost:3000 · Kibana http://localhost:5601 ·
Schema Registry http://localhost:8086/ui

## Current status

Phase 1 (MVP) ✅ and most of Phase 2 ✅ are done (auth, chat, streaming inference,
assets, distributed tracing, central config, schema registry, CQRS search, Helm
charts). Remaining: Testcontainers tests, load testing, and the React frontend.
See [PHASE2_TODO.md](PHASE2_TODO.md) for the live checklist.

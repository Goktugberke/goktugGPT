# api-gateway

> **Edge Streaming Gateway** — tüm dış trafiğin tek giriş noktası.

## Sorumluluklar

- Routing (`/api/v1/auth/**` → identity, `/api/v1/chats/**` → conversation, vs.)
- JWT validation (Keycloak JWKS'i fetch edip cache'ler)
- Token Bucket rate limit (Redis backed) — kullanıcı bazlı veya IP bazlı
- CORS
- SSE/WebSocket pass-through (inference streaming)
- OpenTelemetry trace context propagasyonu (downstream'lere `traceparent` header iletir)

## Tech Stack

- **Spring Cloud Gateway** (Reactive, Netty)
- **Spring Security 6** (OAuth2 Resource Server, Reactive)
- **Resilience4j** (Circuit Breaker, fallback)
- **Micrometer + OTel** → Jaeger / Prometheus

## Port

- `8080` (public)

## Bağımlılıklar

- Keycloak (`http://keycloak:8080/realms/goktuggpt/protocol/openid-connect/certs`)
- Redis (rate limit storage)
- Tüm downstream servisler (routing target)

## TODO (bir sonraki session)

1. **JWT claim → header mapper** ekle: `userId`, `email`, `roles`'u downstream servislere `X-User-Id`, `X-User-Email`, `X-User-Roles` header olarak ilet (gereksiz JWT parse'lamayı önler).
2. **WebSocket route** test et (`spring.cloud.gateway.routes.*.predicates: Header=Upgrade,websocket`).
3. **Per-route rate limit** — `/api/v1/inference/**` için daha düşük limit (pahalı), `/api/v1/chats/**` için yüksek.
4. **CircuitBreaker filter** her route'a ekle (`name: cb`, `args.fallbackUri: forward:/fallback/...`).
5. **Global error handler** — downstream timeout/4xx/5xx için unified JSON error response.
6. **CORS config** — frontend origin'i (`http://localhost:5173`) izin ver.
7. **mTLS** (Faz 3) — service mesh'e (Istio) geçince sertifikasız.

## Çalıştırma (lokal, infra ayağa kalktıktan sonra)

```bash
mvn -pl services/api-gateway -am spring-boot:run
```

veya:

```bash
docker compose up -d --build api-gateway
```

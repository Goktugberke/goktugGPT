# billing-service  [Faz 2]

> **Quota, abonelik, kullanım sayımı.**

## Sorumluluklar

- **Quota check (sync):** kullanıcının token/request kotası kaldı mı? — Redis token bucket
- **Subscription state:** Free / Plus / Pro / Enterprise plan bilgisi
- **Usage recording (async):** her completed inference'da token sayımı düşülür (event consumer)
- **Stripe webhook handler (Faz 3):** payment success → plan upgrade
- **Quota refund:** Saga compensate — failed inference quotayı geri ver

## Mimari Karar — Token Bucket

Her kullanıcının `bucket:user:{userId}` Redis anahtarı:
- **Capacity:** plan'a göre (Free=10/dk, Plus=60/dk, Pro=300/dk)
- **Refill rate:** plan'a göre saniyede X token
- **Lua script** ile atomic check-and-decrement

```lua
-- Pseudo: tokens > 0 ise decrement et + true dön, yoksa false
local tokens = redis.call('GET', KEYS[1]) or 0
if tokens > 0 then
    redis.call('DECR', KEYS[1])
    return 1
end
return 0
```

Bu bucket'a inference orchestrator hot path'te bakar (DB hit yok, <5ms).

Eventual consistency: arka planda `usage.recorded.v1` event'leri DB'ye işler (audit + invoice için).

## Endpoints

| Method | Path | Açıklama |
|--------|------|----------|
| GET | /api/v1/billing/me/quota | Kalan token + reset zamanı |
| GET | /api/v1/billing/me/subscription | Aktif plan + dönem |
| POST | /api/v1/billing/me/subscribe | Stripe checkout session (Faz 3) |
| POST | /internal/billing/check | Orchestrator: quota var mı? (low latency) |
| POST | /internal/billing/refund | Saga compensate: tokenları geri ver |
| POST | /webhooks/stripe | Stripe event handler |

## Eventler

**Subscribe:**
- `user.registered.v1` → free plan oluştur
- `inference.completed.v1` → `tokenUsage` kadar düş, usage_records'a yaz
- `inference.failed.v1` → varsa pre-deduct yapıldıysa refund

**Publish:**
- `usage.recorded.v1` (telemetry için)
- `quota.exceeded.v1` (notification için — Pro plan upgrade önerisi)

## Patterns

- **Token Bucket** (Redis Lua atomic)
- **Event Sourcing (light):** `usage_records` append-only, current quota = derived state
- **CQRS:** Redis = read model (hot path), PostgreSQL = source of truth (audit)
- **Saga compensate participant** (refund endpoint)

## DB Schema (taslak)

```sql
plans                  -- free/plus/pro/enterprise
subscriptions          -- userId → planId, periodStart, periodEnd
usage_records          -- (userId, eventId, promptTokens, completionTokens, occurredAt)
quota_states           -- (userId, periodStart, tokensConsumed, requestsConsumed) — derived
```

## Port

`8085` (internal)

## TODO

1. Spring Boot scaffold (asset-service'i template al)
2. `Plan`, `Subscription`, `UsageRecord` entity'leri + Flyway migration
3. `RedisTokenBucket` (Lua script with `RedisTemplate.execute()`)
4. `BillingController` (public endpoints)
5. `InternalBillingController` (orchestrator için, `X-Internal-Token` header zorunlu)
6. `InferenceCompletedConsumer` → usage düş + Redis bucket sync
7. `UserRegisteredConsumer` → free plan subscription oluştur
8. **Stripe entegrasyonu Faz 3'e** ertelendi.

## Faz Önceliği

**Faz 2.** Faz 1 MVP'de orchestrator quota check'i bypass eder (herkes unlimited).

# Event Schema Registry

> **Tek doğruluk kaynağı.** Her servis Kafka'ya event publish ederken / consume ederken bu klasördeki şemaya uymak zorunda.

## Naming Convention

`<domain>.<event-name>.v<version>.json`

- `domain` — bounded context kısa adı (`user`, `chat`, `message`, `inference`, `asset`, `billing`)
- `event-name` — geçmiş zaman fiili (`registered`, `created`, `completed`)
- `version` — major version. Breaking change → yeni dosya `v2`. Eski tüketici hala `v1`'i okur.

## Topic Mapping

| Topic | Schema |
|-------|--------|
| `user.events`         | `user.registered.v1`, `user.profile-updated.v1`, `user.deleted.v1` |
| `chat.events`         | `chat.created.v1`, `chat.deleted.v1`, `chat.title-changed.v1` |
| `message.events`      | `message.user-sent.v1`, `message.assistant-completed.v1` |
| `inference.events`    | `inference.started.v1`, `inference.completed.v1`, `inference.failed.v1` |
| `asset.events`        | `asset.uploaded.v1`, `asset.deleted.v1` |
| `billing.events`      | `usage.recorded.v1`, `quota.exceeded.v1` |

## Envelope (Tüm Eventler İçin Ortak)

```json
{
  "eventId":    "uuid",
  "eventType":  "message.user-sent.v1",
  "occurredAt": "2026-05-04T12:34:56.789Z",
  "producer":   "conversation-service",
  "traceId":    "string",
  "userId":     "string",
  "payload":    { ... }
}
```

`payload` — event'e özel alan (aşağıdaki şemalar bunu tanımlar).

## Şemaları Düzenlerken

- **Asla** mevcut alanı silme veya tipini değiştirme. Yeni alan ekleyebilirsin (additive).
- Breaking change gerekirse → yeni dosya `v2.json`, eski şema `v1.json` korunur.
- Producer önce yeni versiyona geçer (her iki version'a yazar), tüm consumer'lar `v2`'ye geçince `v1` deprecate edilir.

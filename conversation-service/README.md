# conversation-service

> **Sohbet ve mesaj yönetimi.** Saritaygpt'taki monolit ChatService + ChatRepository + AzureDataLakeService karmaşasının yerine, sadece persistence + event publishing'e odaklanan tek sorumluluk.

## Sorumluluklar

- Chat CRUD (oluştur, listele, getir, güncelle, sil)
- Message CRUD (kullanıcı mesajını persist, AI response'u persist)
- Geçmiş arama (Elasticsearch — CQRS read model)
- Pagination (cursor-based)
- `message.user-sent.v1` event publish (Outbox pattern)
- `inference.completed.v1` consume → assistant message persist (Saga step)

## Mimari Karar — NEDEN persistence ile inference'ı ayırdık?

Saritaygpt'ta `ChatService.streamUserMessageWithFiles()` aynı thread'de:
1. Chat oluştur (DB)
2. File upload (Azure)
3. Agent service çağır (HTTP)
4. Stream'i frontend'e relay et (SSE)
5. Response'u persist (DB)

→ Tek bir thread çöktüğünde her şey çöküyor, scaling imkansız, transactional boundary yanlış.

Yeni mimari:
- **conversation-service** sadece DB + event yapar
- **inference-orchestrator** Saga'yı yönetir
- **asset-service** dosyaları yönetir
- Hepsi event'lerle (Kafka) gevşek bağlı

## Patterns Implemented

### 1. Transactional Outbox

`OutboxEntity` + `OutboxPoller` — DB ve Kafka arasında atomic publish garantisi.

```java
@Transactional
public Message saveUserMessage(...) {
    Message m = messageRepo.save(...);

    OutboxEntity event = OutboxEntity.builder()
        .aggregateId(m.getId())
        .eventType("message.user-sent.v1")
        .topic("message.events")
        .payload(buildPayload(m))
        .build();
    outboxRepo.save(event);   // SAME TRANSACTION

    return m;
}
// Commit'ten sonra @Scheduled OutboxPoller bu satırı Kafka'ya basar
```

### 2. Idempotency

`IdempotencyEntity` + `IdempotencyFilter` — POST endpointleri için.

Frontend `X-Idempotency-Key: <uuid>` gönderir. Aynı key + endpoint + userId tekrar geldiyse → cached response. Test için Postman'da retry → duplicate mesaj oluşturmaz.

### 3. CQRS

- **Write side:** PostgreSQL (`chats`, `messages`)
- **Read side:** Elasticsearch (`chat-search` index)
- `chat.created.v1` / `chat.title-changed.v1` event'leri consume eden async indexer (henüz yazılmadı — TODO)

`GET /api/v1/chats/search?q=foo` → Elasticsearch'e gider.

### 4. Saga Participation

Inference orchestrator `inference.completed.v1` event publish ettiğinde, conversation-service consumer bunu yakalar ve assistant message row'u persist eder. Inference orchestrator bunu sync REST ile yapsaydı: tight coupling + cascade failure riski.

## Endpoints

| Method | Path | Header | Açıklama |
|--------|------|--------|----------|
| POST | /api/v1/chats | | Boş chat oluştur |
| GET | /api/v1/chats?page=&size= | | Listele |
| GET | /api/v1/chats/{chatId} | | Detay (ilk N mesaj) |
| PUT | /api/v1/chats/{chatId} | | Title değiştir |
| DELETE | /api/v1/chats/{chatId} | | |
| POST | /api/v1/chats/{chatId}/messages | X-Idempotency-Key (zorunlu) | User mesajı persist |
| GET | /api/v1/chats/{chatId}/messages?cursor= | | Mesaj geçmişi |
| GET | /api/v1/chats/search?q= | | ES'te title arama |

## Eventler

**Publish (via Outbox):**
- `message.user-sent.v1`
- `chat.created.v1`, `chat.deleted.v1`, `chat.title-changed.v1`

**Subscribe:**
- `inference.completed.v1` → assistant message persist
- `user.deleted.v1` → kullanıcının chat'lerini sil (cascade)

## Port

`8082` (internal)

## TODO (bir sonraki session)

1. **Domain entity'leri:** `ChatEntity`, `MessageEntity`, `ChatRepository`, `MessageRepository`
2. **DTO'lar:** `ChatDto`, `MessageDto`, `CreateMessageRequest`, `MessageResponse`
3. **REST Controller'lar:** `ChatController`, `MessageController`
4. **`IdempotencyAspect` veya `IdempotencyFilter`:** `@Idempotent` annotation veya `X-Idempotency-Key` zorunluluğu
5. **`InferenceCompletedConsumer`:** `@KafkaListener("inference.events")` → assistant message persist
6. **`ChatSearchProjector`:** chat event'lerini Elasticsearch'e index'leyen consumer
7. **OutboxPoller**'a exponential backoff (`attempt_count` × 2^n saniye)
8. **Cleanup job:** expired idempotency keys, soft-deleted chat'ler için TTL
9. **Testler:** Testcontainers (Postgres + Kafka) ile integration test — `OutboxPoller` flow'u garanti

## Çalıştırma

```bash
docker compose --profile infra up -d
mvn -pl services/conversation-service -am spring-boot:run
```

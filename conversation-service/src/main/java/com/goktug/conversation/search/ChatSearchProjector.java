package com.goktug.conversation.search;

import com.fasterxml.jackson.databind.JsonNode;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

import java.time.OffsetDateTime;

/**
 * CQRS read model projector.
 *
 * `chat.events` topic'ini consume eder, ES `chat-search` index'ini günceller.
 * Read endpoint'i (GET /api/v1/chats/search?q=) bu index'e gider.
 *
 * Eventually consistent: write (Postgres) anlık, read (ES) ms-saniye gecikmeli.
 *
 * Idempotent: aynı eventId tekrar gelirse aynı doküman üst yazılır,
 * kayıp/duplicate yok.
 *
 * NOT: ES index'i aşağı düşerse projector backlog biriktirir; consumer
 * lag'i Prometheus metric ile izlenir (kafka_consumer_lag).
 *
 * Initial reindex: tüm chat'leri ES'e basmak için ayrı bir job ekle (Faz 2).
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class ChatSearchProjector {

    private final ChatSearchRepository searchRepository;

    @KafkaListener(topics = "chat.events", groupId = "conversation-service-search-projector")
    public void onChatEvent(JsonNode envelope) {
        String type = text(envelope, "eventType");
        if (type == null) return;

        try {
            JsonNode payload = envelope.get("payload");
            switch (type) {
                case "chat.created.v1" -> upsert(payload, false);
                case "chat.title-changed.v1" -> upsert(payload, false);
                case "chat.deleted.v1" -> markDeleted(payload);
                default -> log.debug("Ignoring chat event type: {}", type);
            }
        } catch (Exception ex) {
            log.error("Failed to project chat event: {}", envelope, ex);
            // TODO: dead letter queue
        }
    }

    private void upsert(JsonNode payload, boolean deleted) {
        String chatId = payload.get("chatId").asText();
        String userId = payload.get("userId").asText();
        String title = payload.has("title") ? payload.get("title").asText() : null;

        // Mevcut dokümanı çek (eğer varsa) — title değişmemiş olabilir
        ChatSearchDocument doc = searchRepository.findById(chatId)
            .orElseGet(() -> ChatSearchDocument.builder()
                .id(chatId)
                .userId(userId)
                .createdAt(OffsetDateTime.now())
                .build());

        if (title != null) doc.setTitle(title);
        doc.setUpdatedAt(OffsetDateTime.now());
        doc.setDeleted(deleted);

        searchRepository.save(doc);
        log.debug("ES upsert chat={} title={}", chatId, title);
    }

    private void markDeleted(JsonNode payload) {
        String chatId = payload.get("chatId").asText();
        searchRepository.findById(chatId).ifPresent(doc -> {
            doc.setDeleted(true);
            doc.setUpdatedAt(OffsetDateTime.now());
            searchRepository.save(doc);
        });
    }

    private static String text(JsonNode n, String f) {
        return n != null && n.has(f) ? n.get(f).asText() : null;
    }
}

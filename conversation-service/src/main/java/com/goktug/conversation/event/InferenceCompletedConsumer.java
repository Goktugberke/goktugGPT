package com.goktug.conversation.event;

import com.fasterxml.jackson.databind.JsonNode;
import com.goktug.conversation.service.MessageService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

import java.util.UUID;

/**
 * Saga step (consumer side):
 *   inference-orchestrator → inference.completed.v1 → bu consumer →
 *     conversation-service.messages tablosuna assistant message persist.
 *
 * Idempotency: aynı eventId tekrar gelirse (Kafka at-least-once retry),
 * messageId zaten DB'de varsa skip et. Şu an basit — TODO: processed_events
 * tablosuyla guard.
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class InferenceCompletedConsumer {

    private final MessageService messageService;

    @KafkaListener(topics = "inference.events", groupId = "conversation-service")
    public void onInferenceEvent(JsonNode envelope) {
        String type = text(envelope, "eventType");
        if (!"inference.completed.v1".equals(type)) {
            log.debug("Ignoring event type: {}", type);
            return;
        }

        try {
            JsonNode payload = envelope.get("payload");
            UUID chatId = UUID.fromString(payload.get("chatId").asText());
            UUID userId = UUID.fromString(payload.get("userId").asText());
            String responseText = payload.get("responseText").asText();
            String modelUsed = payload.has("modelUsed") ? payload.get("modelUsed").asText() : null;

            Integer tokenCount = null;
            if (payload.has("tokenUsage")) {
                JsonNode usage = payload.get("tokenUsage");
                int prompt = usage.has("promptTokens") ? usage.get("promptTokens").asInt() : 0;
                int completion = usage.has("completionTokens") ? usage.get("completionTokens").asInt() : 0;
                tokenCount = prompt + completion;
            }

            messageService.saveAssistantMessage(chatId, userId, responseText, modelUsed, tokenCount, null);
        } catch (Exception ex) {
            log.error("Failed to process inference.completed event: {}", envelope, ex);
            // Şu an: log + drop. Production'da DLQ (dead letter queue) iyi olur.
        }
    }

    private static String text(JsonNode n, String field) {
        return n != null && n.has(field) ? n.get(field).asText() : null;
    }
}

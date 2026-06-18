package com.goktug.inference.event;

import com.fasterxml.jackson.databind.JsonNode;
import com.goktug.inference.saga.InferenceSaga;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

import java.util.UUID;

/**
 * Event-driven Saga trigger.
 *
 * MÄ°MARÄ° KARAR â€” Bu consumer Ä°KÄ° alternatif baÅŸlatma yolundan biri:
 *
 *   YOL 1 (sync):   Frontend â†’ POST /api/v1/inference/stream
 *                   â†’ InferenceController â†’ Saga.execute()
 *                   â†’ SSE stream Frontend'e
 *                   âœ“ Real-time stream
 *                   âœ— Stream koparsa kullanÄ±cÄ±ya cevap gitmez
 *
 *   YOL 2 (async):  Frontend â†’ POST /chats/{id}/messages
 *                   â†’ conversation-service â†’ message.user-sent.v1 event
 *                   â†’ BURASI: orchestrator event consume eder
 *                   â†’ Saga.execute()  (frontend SSE baÄŸlanmasa bile Ã§alÄ±ÅŸÄ±r)
 *                   â†’ inference.completed.v1 event
 *                   â†’ conversation-service assistant message persist
 *                   âœ“ Background reliable execution (network kopsa bile)
 *                   âœ— SSE stream yok â€” frontend long-polling veya
 *                     WebSocket ile sonucu bekler
 *
 * Ä°ki yol da aynÄ± Saga'yÄ± Ã§aÄŸÄ±rÄ±r; fark "kim tetikliyor".
 *
 * Faz 1 davranÄ±ÅŸÄ±: SADECE YOL 1 aktif (frontend SSE bekler).
 * Bu consumer'Ä± Faz 2'de aktive et â€” `inference.event-triggered=true` config ile.
 * Åžu anlÄ±k @KafkaListener autoStartup=false; toggleAble.
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class UserMessageConsumer {

    private final InferenceSaga saga;

    @Value("${inference.event-triggered:false}")
    private boolean eventTriggered;

    @KafkaListener(
        topics = "message.events",
        groupId = "inference-orchestrator-trigger",
        autoStartup = "${inference.event-triggered:false}"
    )
    public void onMessageEvent(JsonNode envelope) {
        if (!eventTriggered) return;

        String type = envelope.has("eventType") ? envelope.get("eventType").asText() : null;
        if (!"message.user-sent.v1".equals(type)) return;

        try {
            JsonNode payload = envelope.get("payload");
            UUID chatId = UUID.fromString(payload.get("chatId").asText());
            UUID userId = UUID.fromString(payload.get("userId").asText());
            UUID messageId = UUID.fromString(payload.get("messageId").asText());
            String text = payload.get("text").asText();
            String modelHint = payload.has("modelHint") && !payload.get("modelHint").isNull()
                ? payload.get("modelHint").asText() : null;

            log.info("Event-triggered saga start: chatId={} messageId={}", chatId, messageId);

            InferenceSaga.InferenceRequest req = new InferenceSaga.InferenceRequest(
                chatId, userId, messageId, text, modelHint);

            // Async execute â€” Flux'Ä± subscribe et, token'larÄ± kaydetme
            // (sonuÃ§ inference.completed.v1 event'i ile yayÄ±lacak).
            saga.execute(req)
                .doOnError(err -> log.error("Async saga failed: {}", err.getMessage()))
                .subscribe();

        } catch (Exception ex) {
            log.error("Failed to process message.user-sent event: {}", envelope, ex);
        }
    }
}


package com.goktug.notification.event;

import com.fasterxml.jackson.databind.JsonNode;
import com.goktug.notification.service.EmailService;
import com.goktug.notification.ws.NotificationWebSocketHandler;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.UUID;

@Component
@RequiredArgsConstructor
@Slf4j
public class NotificationConsumer {

    private final EmailService emailService;
    private final NotificationWebSocketHandler ws;

    @KafkaListener(topics = "user.events", groupId = "notification-service-user")
    public void onUserEvent(JsonNode envelope) {
        String type = envelope.has("eventType") ? envelope.get("eventType").asText() : null;
        if (!"user.registered.v1".equals(type)) return;

        JsonNode payload = envelope.get("payload");
        if (payload == null) return;

        String email = payload.has("email") ? payload.get("email").asText() : null;
        String displayName = payload.has("displayName") ? payload.get("displayName").asText() : "";
        String userIdStr = payload.has("userId") ? payload.get("userId").asText() : null;

        if (email != null) {
            String subject = "goktugGPT'ye hoş geldin!";
            String body = String.format(
                "Merhaba %s,%n%nHesabın başarıyla oluşturuldu. " +
                "Ücretsiz planın aktif — aylık 100.000 token. İyi sohbetler!%n%n— goktugGPT",
                displayName.isBlank() ? "kullanıcı" : displayName
            );
            emailService.send(email, subject, body);
        }

        if (userIdStr != null) {
            try {
                UUID userId = UUID.fromString(userIdStr);
                ws.sendToUser(userId, Map.of(
                    "type", "welcome",
                    "message", "goktugGPT'ye hoş geldin!"
                ));
            } catch (IllegalArgumentException ignored) {}
        }
    }

    @KafkaListener(topics = "inference.events", groupId = "notification-service-inference")
    public void onInferenceEvent(JsonNode envelope) {
        String type = envelope.has("eventType") ? envelope.get("eventType").asText() : null;
        if (type == null) return;

        JsonNode payload = envelope.get("payload");
        if (payload == null) return;

        String userIdStr = payload.has("userId") ? payload.get("userId").asText() : null;
        if (userIdStr == null) return;

        UUID userId;
        try {
            userId = UUID.fromString(userIdStr);
        } catch (IllegalArgumentException e) {
            return;
        }

        switch (type) {
            case "inference.completed.v1" -> ws.sendToUser(userId, Map.of(
                "type", "inference.completed",
                "jobId", payload.path("jobId").asText(),
                "chatId", payload.path("chatId").asText()
            ));
            case "inference.failed.v1" -> ws.sendToUser(userId, Map.of(
                "type", "inference.failed",
                "jobId", payload.path("jobId").asText(),
                "reason", payload.path("reason").asText()
            ));
            default -> log.debug("Ignoring inference event type: {}", type);
        }
    }
}

package com.goktug.billing.event;

import com.fasterxml.jackson.databind.JsonNode;
import com.goktug.billing.service.BillingService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

import java.util.UUID;

@Component
@RequiredArgsConstructor
@Slf4j
public class UserRegisteredConsumer {

    private final BillingService billingService;

    @KafkaListener(topics = "user.events", groupId = "billing-service-user-registered")
    public void onUserEvent(JsonNode envelope) {
        String type = envelope.has("eventType") ? envelope.get("eventType").asText() : null;
        if (!"user.registered.v1".equals(type)) return;

        try {
            JsonNode payload = envelope.get("payload");
            UUID userId = UUID.fromString(payload.get("userId").asText());
            billingService.ensureFreeSubscription(userId);
            log.info("Free plan provisioned for new user={}", userId);
        } catch (Exception ex) {
            log.error("Failed to handle user.registered.v1: {}", envelope, ex);
            throw ex;  // Spring Kafka error handler retries with backoff
        }
    }
}

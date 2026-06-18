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
public class InferenceCompletedConsumer {

    private final BillingService billingService;

    @KafkaListener(topics = "inference.events", groupId = "billing-service-inference")
    public void onInferenceEvent(JsonNode envelope) {
        String type = envelope.has("eventType") ? envelope.get("eventType").asText() : null;
        if (!"inference.completed.v1".equals(type)) return;

        try {
            JsonNode payload = envelope.get("payload");
            UUID userId = UUID.fromString(payload.get("userId").asText());
            UUID jobId = payload.has("jobId") ? UUID.fromString(payload.get("jobId").asText()) : null;
            String model = payload.has("modelUsed") ? payload.get("modelUsed").asText() : null;

            int promptTokens = 0, completionTokens = 0;
            JsonNode usage = payload.get("tokenUsage");
            if (usage != null) {
                promptTokens = usage.has("promptTokens") ? usage.get("promptTokens").asInt() : 0;
                completionTokens = usage.has("completionTokens") ? usage.get("completionTokens").asInt() : 0;
            }
            Integer latencyMs = payload.has("latencyMs") ? payload.get("latencyMs").asInt() : null;

            billingService.recordUsage(userId, jobId, model, promptTokens, completionTokens, latencyMs);
        } catch (Exception ex) {
            log.error("Failed to handle inference.completed.v1: {}", envelope, ex);
            throw ex;
        }
    }
}

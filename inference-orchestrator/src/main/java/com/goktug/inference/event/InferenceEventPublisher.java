package com.goktug.inference.event;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Component;

import java.time.OffsetDateTime;
import java.util.Map;
import java.util.UUID;

/**
 * Kafka event publisher for inference.events topic.
 *
 * NOT: inference-orchestrator burada Outbox pattern KULLANMIYOR Ã§Ã¼nkÃ¼
 * orchestrator stateful saga'yÄ± zaten DB'de tutar; final completion
 * event'i side-effect olarak basitÃ§e basabilir. EÄŸer event'in mutlaka
 * gitmesi gerekiyorsa Outbox eklenir (TODO Faz 2).
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class InferenceEventPublisher {

    private static final String TOPIC = "inference.events";

    private final KafkaTemplate<String, Object> kafkaTemplate;
    private final ObjectMapper objectMapper;

    public void publishCompleted(
            UUID jobId, UUID chatId, UUID userId, UUID userMessageId,
            String responseText, String modelUsed,
            int promptTokens, int completionTokens, long latencyMs
    ) {
        ObjectNode payload = objectMapper.createObjectNode();
        payload.put("jobId", jobId.toString());
        payload.put("chatId", chatId.toString());
        payload.put("userId", userId.toString());
        payload.put("messageId", userMessageId.toString());
        payload.put("responseText", responseText);
        if (modelUsed != null) payload.put("modelUsed", modelUsed);

        ObjectNode usage = objectMapper.createObjectNode();
        usage.put("promptTokens", promptTokens);
        usage.put("completionTokens", completionTokens);
        payload.set("tokenUsage", usage);

        payload.put("latencyMs", latencyMs);
        payload.put("completedAt", OffsetDateTime.now().toString());

        publish(jobId, "inference.completed.v1", payload);
    }

    public void publishFailed(UUID jobId, UUID chatId, UUID userId, String reason) {
        ObjectNode payload = objectMapper.createObjectNode();
        payload.put("jobId", jobId.toString());
        payload.put("chatId", chatId.toString());
        payload.put("userId", userId.toString());
        payload.put("reason", reason);
        payload.put("failedAt", OffsetDateTime.now().toString());
        publish(jobId, "inference.failed.v1", payload);
    }

    private void publish(UUID jobId, String eventType, ObjectNode payload) {
        ObjectNode envelope = objectMapper.createObjectNode();
        envelope.put("eventId", UUID.randomUUID().toString());
        envelope.put("eventType", eventType);
        envelope.put("occurredAt", OffsetDateTime.now().toString());
        envelope.put("producer", "inference-orchestrator");
        envelope.set("payload", payload);

        kafkaTemplate.send(TOPIC, jobId.toString(), envelope)
            .whenComplete((res, ex) -> {
                if (ex != null) log.error("Failed to publish {}: {}", eventType, ex.getMessage());
                else log.debug("Published {} for job {}", eventType, jobId);
            });
    }
}


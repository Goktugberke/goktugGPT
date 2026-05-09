package com.goktug.conversation.outbox;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.time.OffsetDateTime;
import java.util.UUID;

/**
 * Outbox helper — domain logic'in event yayınlamak için kullandığı yüzey.
 *
 * Domain service şöyle kullanır (TEK transaction içinde):
 *
 *     ChatEntity chat = chatRepo.save(chat);
 *     outboxPublisher.publish(
 *         "chat.events",
 *         "chat.created.v1",
 *         chat.getId(),
 *         payload,
 *         userId,
 *         traceId
 *     );
 *
 * Insert outbox tablosuna gider — commit sonrası OutboxPoller Kafka'ya basar.
 */
@Component
@RequiredArgsConstructor
public class OutboxPublisher {

    private final OutboxRepository repository;
    private final ObjectMapper objectMapper;

    public OutboxEntity publish(
            String topic,
            String eventType,
            UUID aggregateId,
            Object payload,
            UUID userId,
            String traceId
    ) {
        ObjectNode envelope = objectMapper.createObjectNode();
        envelope.put("eventId", UUID.randomUUID().toString());
        envelope.put("eventType", eventType);
        envelope.put("occurredAt", OffsetDateTime.now().toString());
        envelope.put("producer", "conversation-service");
        if (traceId != null) envelope.put("traceId", traceId);
        if (userId != null) envelope.put("userId", userId.toString());
        envelope.set("payload", objectMapper.valueToTree(payload));

        ObjectNode headers = objectMapper.createObjectNode();
        if (traceId != null) headers.put("traceparent", traceId);

        OutboxEntity entity = OutboxEntity.builder()
            .aggregateId(aggregateId)
            .eventType(eventType)
            .topic(topic)
            .payload((JsonNode) envelope)
            .headers((JsonNode) headers)
            .build();
        return repository.save(entity);
    }
}

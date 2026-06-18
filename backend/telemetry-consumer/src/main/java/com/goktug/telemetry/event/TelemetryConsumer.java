package com.goktug.telemetry.event;

import com.fasterxml.jackson.databind.JsonNode;
import com.goktug.telemetry.sink.ColdStorageSink;
import com.goktug.telemetry.sink.ElasticsearchSink;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

/**
 * Fans every domain event into two sinks:
 *   - Elasticsearch  (hot, queryable analytics — last 30d)
 *   - Cold storage   (S3/MinIO ndjson — feeds RLHF / batch jobs)
 *
 * Each topic listens with its own consumer group so re-deploying telemetry
 * does not affect upstream services. Failures are logged but never re-thrown;
 * downstream sinks are independent and one being down should not stop the other.
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class TelemetryConsumer {

    private final ElasticsearchSink es;
    private final ColdStorageSink cold;

    @KafkaListener(topics = "inference.events", groupId = "telemetry-consumer-inference")
    public void onInference(JsonNode envelope) {
        record("inference", envelope);
    }

    @KafkaListener(topics = "message.events", groupId = "telemetry-consumer-message")
    public void onMessage(JsonNode envelope) {
        record("message", envelope);
    }

    @KafkaListener(topics = "user.events", groupId = "telemetry-consumer-user")
    public void onUser(JsonNode envelope) {
        record("user", envelope);
    }

    @KafkaListener(topics = "chat.events", groupId = "telemetry-consumer-chat")
    public void onChat(JsonNode envelope) {
        record("chat", envelope);
    }

    @KafkaListener(topics = "asset.events", groupId = "telemetry-consumer-asset")
    public void onAsset(JsonNode envelope) {
        record("asset", envelope);
    }

    private void record(String category, JsonNode envelope) {
        String type = envelope != null && envelope.has("eventType")
            ? envelope.get("eventType").asText() : "unknown";
        try {
            es.index(category, type, envelope);
        } catch (Exception e) {
            log.warn("ES sink failed (type={}): {}", type, e.getMessage());
        }
        try {
            cold.append(category, envelope);
        } catch (Exception e) {
            log.warn("Cold sink failed (type={}): {}", type, e.getMessage());
        }
    }
}

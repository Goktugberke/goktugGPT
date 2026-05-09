package com.goktug.conversation.outbox;

import com.fasterxml.jackson.databind.JsonNode;
import io.hypersistence.utils.hibernate.type.json.JsonBinaryType;
import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.Type;

import java.time.OffsetDateTime;
import java.util.UUID;

/**
 * Transactional Outbox row.
 *
 * Domain operation (Ã¶rn: yeni mesaj yazma) sÄ±rasÄ±nda ilgili event bu tabloya
 * AYNI transaction'da insert edilir. BÃ¶ylece DB commit edilmeden Kafka'ya
 * birÅŸey gitmez ve "DB yazÄ±ldÄ± ama event gitmedi" durumu oluÅŸmaz.
 *
 * AyrÄ± bir scheduler (OutboxPoller) `processed_at IS NULL` olan satÄ±rlarÄ±
 * okur, Kafka'ya publish eder, sonra `processed_at` set eder.
 *
 * At-least-once garantili â†’ consumer tarafta idempotent olmak ÅŸart
 * (eventId tekrar gÃ¶rÃ¼lÃ¼rse skip).
 */
@Entity
@Table(name = "outbox")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class OutboxEntity {

    @Id
    @GeneratedValue
    private UUID id;

    @Column(name = "aggregate_id", nullable = false)
    private UUID aggregateId;

    @Column(name = "event_type", nullable = false, length = 120)
    private String eventType;

    @Column(nullable = false, length = 120)
    private String topic;

    @Type(JsonBinaryType.class)
    @Column(columnDefinition = "jsonb", nullable = false)
    private JsonNode payload;

    @Type(JsonBinaryType.class)
    @Column(columnDefinition = "jsonb")
    private JsonNode headers;

    @Column(name = "created_at", nullable = false)
    private OffsetDateTime createdAt;

    @Column(name = "processed_at")
    private OffsetDateTime processedAt;

    @Column(name = "attempt_count", nullable = false)
    private int attemptCount;

    @Column(name = "last_error", columnDefinition = "TEXT")
    private String lastError;

    @PrePersist
    void onCreate() {
        if (createdAt == null) createdAt = OffsetDateTime.now();
    }
}



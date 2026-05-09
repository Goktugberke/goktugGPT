package com.goktug.conversation.idempotency;

import com.fasterxml.jackson.databind.JsonNode;
import io.hypersistence.utils.hibernate.type.json.JsonBinaryType;
import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.Type;

import java.time.OffsetDateTime;
import java.util.UUID;

/**
 * Idempotency Pattern.
 *
 * Frontend retry yaparken (Ã¶rn: network kesildi, mesaj gitmedi sandÄ±)
 * aynÄ± POST tekrar gelirse duplicate kayÄ±t oluÅŸmamasÄ± iÃ§in.
 *
 * AkÄ±ÅŸ:
 * 1. POST geldi â†’ user X-Idempotency-Key header gÃ¶nderdi
 * 2. (userId, key, endpoint) tuple'Ä± bu tabloda var mÄ±?
 * - VAR â†’ kayÄ±tlÄ± response'u dÃ¶n (yeniden Ã§alÄ±ÅŸtÄ±rma)
 * - YOK â†’ iÅŸlemi yap, response'u kaydet, dÃ¶n
 * 3. expires_at'e gÃ¶re TTL cleanup (job veya partial index)
 */
@Entity
@Table(name = "idempotency_keys", uniqueConstraints = @UniqueConstraint(columnNames = { "user_id", "idempotency_key",
        "endpoint" }))
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class IdempotencyEntity {

    @Id
    @GeneratedValue
    private UUID id;

    @Column(name = "user_id", nullable = false)
    private UUID userId;

    @Column(name = "idempotency_key", nullable = false, length = 120)
    private String idempotencyKey;

    @Column(nullable = false, length = 120)
    private String endpoint;

    @Type(JsonBinaryType.class)
    @Column(name = "response_body", columnDefinition = "jsonb")
    private JsonNode responseBody;

    @Column(name = "response_status")
    private Integer responseStatus;

    @Column(name = "created_at", nullable = false)
    private OffsetDateTime createdAt;

    @Column(name = "expires_at", nullable = false)
    private OffsetDateTime expiresAt;

    @PrePersist
    void onCreate() {
        if (createdAt == null)
            createdAt = OffsetDateTime.now();
        if (expiresAt == null)
            expiresAt = createdAt.plusHours(24);
    }
}



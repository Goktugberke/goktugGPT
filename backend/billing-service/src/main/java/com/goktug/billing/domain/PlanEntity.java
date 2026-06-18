package com.goktug.billing.domain;

import jakarta.persistence.*;
import lombok.*;

import java.time.OffsetDateTime;
import java.util.UUID;

@Entity
@Table(name = "plans")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class PlanEntity {

    @Id
    private UUID id;

    @Column(nullable = false, unique = true)
    private String name;

    @Column(name = "monthly_token_quota", nullable = false)
    private long monthlyTokenQuota;

    @Column(name = "rate_limit_capacity", nullable = false)
    private int rateLimitCapacity;

    @Column(name = "rate_limit_refill_per_sec", nullable = false)
    private double rateLimitRefillPerSec;

    @Column(name = "created_at", nullable = false, updatable = false)
    private OffsetDateTime createdAt;

    @PrePersist
    void prePersist() {
        if (createdAt == null) createdAt = OffsetDateTime.now();
        if (id == null) id = UUID.randomUUID();
    }
}

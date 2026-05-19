package com.goktug.billing.domain;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.UUID;

public interface UsageRecordRepository extends JpaRepository<UsageRecordEntity, UUID> {
    boolean existsByJobId(UUID jobId);
}

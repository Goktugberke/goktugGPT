package com.goktug.billing.domain;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.Optional;
import java.util.UUID;

public interface SubscriptionRepository extends JpaRepository<SubscriptionEntity, UUID> {

    Optional<SubscriptionEntity> findByUserId(UUID userId);

    @Modifying
    @Query("UPDATE SubscriptionEntity s SET s.tokensUsed = s.tokensUsed + :delta WHERE s.userId = :userId")
    int incrementTokensUsed(@Param("userId") UUID userId, @Param("delta") long delta);
}

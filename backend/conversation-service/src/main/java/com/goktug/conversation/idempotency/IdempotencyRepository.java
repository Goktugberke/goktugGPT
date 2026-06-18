package com.goktug.conversation.idempotency;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.time.OffsetDateTime;
import java.util.Optional;
import java.util.UUID;

public interface IdempotencyRepository extends JpaRepository<IdempotencyEntity, UUID> {

    Optional<IdempotencyEntity> findByUserIdAndIdempotencyKeyAndEndpoint(
        UUID userId, String idempotencyKey, String endpoint);

    @Modifying
    @Query("DELETE FROM IdempotencyEntity i WHERE i.expiresAt < :now")
    int deleteExpired(@Param("now") OffsetDateTime now);
}

package com.goktug.conversation.domain;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.time.OffsetDateTime;
import java.util.Optional;
import java.util.UUID;

public interface ChatRepository extends JpaRepository<ChatEntity, UUID> {

    Page<ChatEntity> findByUserIdAndDeletedAtIsNullOrderByUpdatedAtDesc(UUID userId, Pageable pageable);

    Optional<ChatEntity> findByIdAndUserIdAndDeletedAtIsNull(UUID id, UUID userId);

    @Modifying
    @Query("UPDATE ChatEntity c SET c.deletedAt = :now WHERE c.id = :id AND c.userId = :userId AND c.deletedAt IS NULL")
    int softDelete(@Param("id") UUID id, @Param("userId") UUID userId, @Param("now") OffsetDateTime now);

    @Modifying
    @Query("UPDATE ChatEntity c SET c.deletedAt = :now WHERE c.userId = :userId AND c.deletedAt IS NULL")
    int softDeleteAllByUser(@Param("userId") UUID userId, @Param("now") OffsetDateTime now);
}

package com.goktug.conversation.domain;

import org.springframework.data.domain.Slice;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.time.OffsetDateTime;
import java.util.UUID;

public interface MessageRepository extends JpaRepository<MessageEntity, UUID> {

    Slice<MessageEntity> findByChatIdAndCreatedAtGreaterThanOrderByCreatedAtAsc(
        UUID chatId, OffsetDateTime cursor, Pageable pageable);

    Slice<MessageEntity> findByChatIdOrderByCreatedAtAsc(UUID chatId, Pageable pageable);

    long countByChatId(UUID chatId);
}

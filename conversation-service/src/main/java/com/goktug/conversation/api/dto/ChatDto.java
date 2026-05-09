package com.goktug.conversation.api.dto;

import com.goktug.conversation.domain.ChatEntity;

import java.time.OffsetDateTime;
import java.util.UUID;

public record ChatDto(
    UUID id,
    String title,
    OffsetDateTime createdAt,
    OffsetDateTime updatedAt
) {
    public static ChatDto from(ChatEntity entity) {
        return new ChatDto(
            entity.getId(),
            entity.getTitle(),
            entity.getCreatedAt(),
            entity.getUpdatedAt()
        );
    }
}

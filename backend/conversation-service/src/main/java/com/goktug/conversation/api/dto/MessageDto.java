package com.goktug.conversation.api.dto;

import com.fasterxml.jackson.databind.JsonNode;
import com.goktug.conversation.domain.MessageEntity;
import com.goktug.conversation.domain.MessageSender;

import java.time.OffsetDateTime;
import java.util.UUID;

public record MessageDto(
    UUID id,
    UUID chatId,
    MessageSender sender,
    String content,
    String modelUsed,
    Integer tokenCount,
    JsonNode attachments,
    OffsetDateTime createdAt
) {
    public static MessageDto from(MessageEntity m) {
        return new MessageDto(
            m.getId(),
            m.getChatId(),
            m.getSender(),
            m.getContent(),
            m.getModelUsed(),
            m.getTokenCount(),
            m.getAttachments(),
            m.getCreatedAt()
        );
    }
}

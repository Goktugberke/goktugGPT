package com.goktug.conversation.service;

import com.goktug.conversation.api.dto.ChatDto;
import com.goktug.conversation.api.dto.CreateChatRequest;
import com.goktug.conversation.api.dto.PageResponse;
import com.goktug.conversation.api.dto.UpdateChatRequest;
import com.goktug.conversation.domain.ChatEntity;
import com.goktug.conversation.domain.ChatRepository;
import com.goktug.conversation.domain.MessageEntity;
import com.goktug.conversation.domain.MessageRepository;
import com.goktug.conversation.error.NotFoundException;
import com.goktug.conversation.outbox.OutboxPublisher;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Slice;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.OffsetDateTime;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Service
@RequiredArgsConstructor
public class ChatService {

    private static final String CHAT_TOPIC = "chat.events";

    private final ChatRepository chatRepository;
    private final MessageRepository messageRepository;
    private final OutboxPublisher outboxPublisher;

    @Transactional
    public ChatDto create(UUID userId, CreateChatRequest req, String traceId) {
        ChatEntity chat = ChatEntity.builder()
            .userId(userId)
            .title(req.title() == null || req.title().isBlank() ? "New chat" : req.title())
            .build();
        chat = chatRepository.save(chat);

        outboxPublisher.publish(
            CHAT_TOPIC,
            "chat.created.v1",
            chat.getId(),
            Map.of(
                "chatId", chat.getId().toString(),
                "userId", userId.toString(),
                "title", chat.getTitle(),
                "createdAt", chat.getCreatedAt().toString()
            ),
            userId,
            traceId
        );

        return ChatDto.from(chat);
    }

    @Transactional(readOnly = true)
    public PageResponse<ChatDto> list(UUID userId, int page, int size) {
        Page<ChatEntity> p = chatRepository.findByUserIdAndDeletedAtIsNullOrderByUpdatedAtDesc(
            userId, PageRequest.of(page, Math.min(size, 100)));
        return PageResponse.from(p, ChatDto::from);
    }

    @Transactional(readOnly = true)
    public ChatDto get(UUID userId, UUID chatId) {
        ChatEntity chat = chatRepository.findByIdAndUserIdAndDeletedAtIsNull(chatId, userId)
            .orElseThrow(() -> new NotFoundException("Chat not found"));
        return ChatDto.from(chat);
    }

    @Transactional(readOnly = true)
    public List<MessageEntity> getMessages(UUID userId, UUID chatId, int size) {
        // Sahiplik kontrolü
        chatRepository.findByIdAndUserIdAndDeletedAtIsNull(chatId, userId)
            .orElseThrow(() -> new NotFoundException("Chat not found"));

        Slice<MessageEntity> slice = messageRepository.findByChatIdOrderByCreatedAtAsc(
            chatId, PageRequest.of(0, Math.min(size, 200)));
        return slice.getContent();
    }

    @Transactional
    public ChatDto rename(UUID userId, UUID chatId, UpdateChatRequest req, String traceId) {
        ChatEntity chat = chatRepository.findByIdAndUserIdAndDeletedAtIsNull(chatId, userId)
            .orElseThrow(() -> new NotFoundException("Chat not found"));
        chat.setTitle(req.title());

        outboxPublisher.publish(
            CHAT_TOPIC,
            "chat.title-changed.v1",
            chat.getId(),
            Map.of(
                "chatId", chat.getId().toString(),
                "userId", userId.toString(),
                "title", chat.getTitle()
            ),
            userId,
            traceId
        );

        return ChatDto.from(chat);
    }

    @Transactional
    public void delete(UUID userId, UUID chatId, String traceId) {
        OffsetDateTime now = OffsetDateTime.now();
        int deleted = chatRepository.softDelete(chatId, userId, now);
        if (deleted == 0) throw new NotFoundException("Chat not found");

        outboxPublisher.publish(
            CHAT_TOPIC,
            "chat.deleted.v1",
            chatId,
            Map.of(
                "chatId", chatId.toString(),
                "userId", userId.toString(),
                "deletedAt", now.toString()
            ),
            userId,
            traceId
        );
    }
}

package com.goktug.conversation.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.goktug.conversation.api.dto.CreateMessageRequest;
import com.goktug.conversation.api.dto.MessageDto;
import com.goktug.conversation.domain.*;
import com.goktug.conversation.error.NotFoundException;
import com.goktug.conversation.outbox.OutboxPublisher;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Slice;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.OffsetDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Service
@RequiredArgsConstructor
@Slf4j
public class MessageService {

    private static final String MESSAGE_TOPIC = "message.events";

    private final ChatRepository chatRepository;
    private final MessageRepository messageRepository;
    private final OutboxPublisher outboxPublisher;

    /**
     * Kullanıcı mesajı persist et + message.user-sent.v1 event'i Outbox'a yaz.
     *
     * Aynı transaction'da hem mesajı hem outbox'ı kaydediyoruz —
     * Transactional Outbox pattern'in özü.
     */
    @Transactional
    public MessageDto saveUserMessage(
            UUID userId,
            UUID chatId,
            String idempotencyKey,
            CreateMessageRequest req,
            String traceId
    ) {
        ChatEntity chat = chatRepository.findByIdAndUserIdAndDeletedAtIsNull(chatId, userId)
            .orElseThrow(() -> new NotFoundException("Chat not found"));

        MessageEntity message = MessageEntity.builder()
            .chatId(chat.getId())
            .sender(MessageSender.USER)
            .content(req.content())
            .attachments(req.attachments())
            .build();
        message = messageRepository.save(message);

        // İlk mesajda chat title'ı otomatik olarak prompt'un ilk kelimelerinden ata
        if (chat.getTitle() == null || "New chat".equals(chat.getTitle())) {
            chat.setTitle(autoTitle(req.content()));
        }
        // touch updatedAt
        chat.setUpdatedAt(OffsetDateTime.now());

        // Event payload — message.user-sent.v1 schema'sına uygun
        Map<String, Object> payload = new HashMap<>();
        payload.put("messageId", message.getId().toString());
        payload.put("chatId", chat.getId().toString());
        payload.put("userId", userId.toString());
        payload.put("text", req.content());
        payload.put("modelHint", req.modelHint());
        payload.put("attachmentIds", List.of());   // TODO: req.attachments'tan asset ID'leri çıkar
        payload.put("idempotencyKey", idempotencyKey);
        payload.put("createdAt", message.getCreatedAt().toString());

        outboxPublisher.publish(
            MESSAGE_TOPIC,
            "message.user-sent.v1",
            message.getId(),
            payload,
            userId,
            traceId
        );

        log.info("User message saved: chatId={} messageId={}", chat.getId(), message.getId());
        return MessageDto.from(message);
    }

    /**
     * AI cevabı persist et — inference-orchestrator'dan gelen
     * inference.completed.v1 event'i tarafından çağrılır (Saga step).
     */
    @Transactional
    public void saveAssistantMessage(
            UUID chatId,
            UUID userId,
            String content,
            String modelUsed,
            Integer tokenCount,
            JsonNode attachments
    ) {
        ChatEntity chat = chatRepository.findById(chatId)
            .orElseThrow(() -> new NotFoundException("Chat " + chatId + " not found"));

        if (!chat.getUserId().equals(userId)) {
            log.warn("Assistant message userId mismatch: chat={} userId={} eventUserId={}",
                chatId, chat.getUserId(), userId);
            // No-op — event'i drop et (security)
            return;
        }

        MessageEntity message = MessageEntity.builder()
            .chatId(chatId)
            .sender(MessageSender.ASSISTANT)
            .content(content)
            .modelUsed(modelUsed)
            .tokenCount(tokenCount)
            .attachments(attachments)
            .build();
        messageRepository.save(message);
        chat.setUpdatedAt(OffsetDateTime.now());

        log.info("Assistant message persisted: chatId={} messageId={} model={}",
            chatId, message.getId(), modelUsed);
    }

    @Transactional(readOnly = true)
    public List<MessageDto> listForChat(UUID userId, UUID chatId, int size, OffsetDateTime cursor) {
        chatRepository.findByIdAndUserIdAndDeletedAtIsNull(chatId, userId)
            .orElseThrow(() -> new NotFoundException("Chat not found"));

        Slice<MessageEntity> slice;
        if (cursor != null) {
            slice = messageRepository.findByChatIdAndCreatedAtGreaterThanOrderByCreatedAtAsc(
                chatId, cursor, PageRequest.of(0, Math.min(size, 200)));
        } else {
            slice = messageRepository.findByChatIdOrderByCreatedAtAsc(
                chatId, PageRequest.of(0, Math.min(size, 200)));
        }
        return slice.getContent().stream().map(MessageDto::from).toList();
    }

    private String autoTitle(String content) {
        String trimmed = content.strip();
        if (trimmed.length() <= 60) return trimmed;
        return trimmed.substring(0, 57) + "...";
    }
}

package com.goktug.conversation.api;

import com.goktug.conversation.api.dto.CreateMessageRequest;
import com.goktug.conversation.api.dto.MessageDto;
import com.goktug.conversation.config.AuthContext;
import com.goktug.conversation.idempotency.IdempotencyService;
import com.goktug.conversation.service.MessageService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

import java.time.OffsetDateTime;
import java.util.List;
import java.util.UUID;

@RestController
@RequestMapping("/api/v1/chats/{chatId}/messages")
@RequiredArgsConstructor
public class MessageController {

    private static final String IDEMPOTENCY_HEADER = "X-Idempotency-Key";

    private final MessageService messageService;
    private final IdempotencyService idempotencyService;

    @PostMapping
    public ResponseEntity<MessageDto> sendUserMessage(
            @PathVariable UUID chatId,
            @RequestBody @Valid CreateMessageRequest body,
            @RequestHeader(value = IDEMPOTENCY_HEADER, required = false) String idempotencyKey,
            HttpServletRequest request) {

        UUID userId = AuthContext.requireUserId(request);

        if (idempotencyKey == null || idempotencyKey.isBlank()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST,
                IDEMPOTENCY_HEADER + " header is required for this endpoint");
        }

        String traceId = request.getHeader("traceparent");
        String endpoint = "POST /api/v1/chats/{chatId}/messages";

        MessageDto dto = idempotencyService.executeIdempotent(
            userId, idempotencyKey, endpoint,
            () -> {
                MessageDto m = messageService.saveUserMessage(userId, chatId, idempotencyKey, body, traceId);
                return new IdempotencyService.ResultWithStatus<>(HttpStatus.CREATED.value(), m);
            }
        );

        return ResponseEntity.status(HttpStatus.CREATED).body(dto);
    }

    @GetMapping
    public List<MessageDto> list(
            @PathVariable UUID chatId,
            @RequestParam(defaultValue = "100") int size,
            @RequestParam(required = false) OffsetDateTime cursor,
            HttpServletRequest request) {
        UUID userId = AuthContext.requireUserId(request);
        return messageService.listForChat(userId, chatId, size, cursor);
    }
}

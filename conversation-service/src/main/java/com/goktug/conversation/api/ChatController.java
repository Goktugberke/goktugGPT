package com.goktug.conversation.api;

import com.goktug.conversation.api.dto.*;
import com.goktug.conversation.config.AuthContext;
import com.goktug.conversation.service.ChatService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.UUID;

@RestController
@RequestMapping("/api/v1/chats")
@RequiredArgsConstructor
public class ChatController {

    private final ChatService chatService;

    @PostMapping
    public ResponseEntity<ChatDto> create(
            @RequestBody(required = false) @Valid CreateChatRequest body,
            HttpServletRequest request) {
        UUID userId = AuthContext.requireUserId(request);
        String traceId = request.getHeader("traceparent");
        ChatDto dto = chatService.create(userId, body == null ? new CreateChatRequest(null) : body, traceId);
        return ResponseEntity.status(HttpStatus.CREATED).body(dto);
    }

    @GetMapping
    public PageResponse<ChatDto> list(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size,
            HttpServletRequest request) {
        UUID userId = AuthContext.requireUserId(request);
        return chatService.list(userId, page, size);
    }

    @GetMapping("/{chatId}")
    public ChatDto get(@PathVariable UUID chatId, HttpServletRequest request) {
        UUID userId = AuthContext.requireUserId(request);
        return chatService.get(userId, chatId);
    }

    @PutMapping("/{chatId}")
    public ChatDto rename(
            @PathVariable UUID chatId,
            @RequestBody @Valid UpdateChatRequest body,
            HttpServletRequest request) {
        UUID userId = AuthContext.requireUserId(request);
        String traceId = request.getHeader("traceparent");
        return chatService.rename(userId, chatId, body, traceId);
    }

    @DeleteMapping("/{chatId}")
    public ResponseEntity<Void> delete(@PathVariable UUID chatId, HttpServletRequest request) {
        UUID userId = AuthContext.requireUserId(request);
        String traceId = request.getHeader("traceparent");
        chatService.delete(userId, chatId, traceId);
        return ResponseEntity.noContent().build();
    }
}

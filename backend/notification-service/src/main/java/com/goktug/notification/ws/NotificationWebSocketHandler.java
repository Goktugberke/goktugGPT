package com.goktug.notification.ws;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import java.io.IOException;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Per-user WebSocket session registry. Routes notifications addressed to
 * a userId to any open WebSocket session(s) for that user.
 *
 * Client connects with: ws://gateway/ws/notifications?userId={uuid}
 * (api-gateway should populate this from JWT before proxying — done by
 * JwtToHeadersFilter on REST, will need a parallel WS filter in Faz 2.)
 *
 * For Faz 1 dev the client passes userId directly as a query param.
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class NotificationWebSocketHandler extends TextWebSocketHandler {

    private final ObjectMapper objectMapper;
    private final Map<UUID, Set<WebSocketSession>> sessions = new ConcurrentHashMap<>();

    @Override
    public void afterConnectionEstablished(WebSocketSession session) {
        UUID userId = extractUserId(session);
        if (userId == null) {
            try { session.close(CloseStatus.BAD_DATA); } catch (IOException ignored) {}
            return;
        }
        sessions.computeIfAbsent(userId, k -> ConcurrentHashMap.newKeySet()).add(session);
        log.info("WS connect user={} session={} totalForUser={}",
            userId, session.getId(), sessions.get(userId).size());
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) {
        UUID userId = extractUserId(session);
        if (userId == null) return;
        Set<WebSocketSession> userSessions = sessions.get(userId);
        if (userSessions != null) {
            userSessions.remove(session);
            if (userSessions.isEmpty()) sessions.remove(userId);
        }
        log.info("WS disconnect user={} session={} status={}", userId, session.getId(), status);
    }

    public void sendToUser(UUID userId, Object payload) {
        Set<WebSocketSession> userSessions = sessions.get(userId);
        if (userSessions == null || userSessions.isEmpty()) {
            log.debug("No active WS session for user={}, skipping push", userId);
            return;
        }
        String json;
        try {
            json = objectMapper.writeValueAsString(payload);
        } catch (Exception e) {
            log.warn("Failed to serialize notification: {}", e.getMessage());
            return;
        }
        TextMessage msg = new TextMessage(json);
        for (WebSocketSession s : userSessions) {
            if (!s.isOpen()) continue;
            try {
                s.sendMessage(msg);
            } catch (IOException e) {
                log.warn("WS send failed for session {}: {}", s.getId(), e.getMessage());
            }
        }
    }

    public int activeUsers() {
        return sessions.size();
    }

    private UUID extractUserId(WebSocketSession session) {
        // 1) Header set by api-gateway JwtToHeadersFilter
        String header = session.getHandshakeHeaders().getFirst("X-User-Id");
        if (header != null) {
            try { return UUID.fromString(header); } catch (IllegalArgumentException ignored) {}
        }
        // 2) Query param fallback for dev
        String query = session.getUri() != null ? session.getUri().getQuery() : null;
        if (query != null) {
            for (String part : query.split("&")) {
                if (part.startsWith("userId=")) {
                    try { return UUID.fromString(part.substring("userId=".length())); }
                    catch (IllegalArgumentException ignored) {}
                }
            }
        }
        return null;
    }
}

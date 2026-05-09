package com.goktug.inference.api;

import com.goktug.inference.saga.InferenceSaga;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;
import org.springframework.http.server.reactive.ServerHttpRequest;
import reactor.core.publisher.Flux;

import java.time.Duration;
import java.util.UUID;

/**
 * Inference SSE endpoint. Frontend bu endpoint'e POST atar (gateway Ã¼zerinden),
 * dÃ¶nen Flux<ServerSentEvent>'i Server-Sent Event olarak alÄ±r.
 *
 * Saga adÄ±mlarÄ± boyunca farklÄ± tipte event'ler gelir:
 *   data: {"type":"token","content":"merhaba"}
 *   data: {"type":"done","tokenCount":42}
 *   data: {"type":"error","content":"reason","terminalState":"BLOCKED"}
 */
@RestController
@RequestMapping("/api/v1/inference")
@RequiredArgsConstructor
@Slf4j
public class InferenceController {

    private final InferenceSaga saga;

    @PostMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<ServerSentEvent<InferenceSaga.TokenEvent>> stream(
            @RequestBody @Valid StreamRequest body,
            ServerHttpRequest request) {

        UUID userId = userIdFromRequest(request);

        InferenceSaga.InferenceRequest req = new InferenceSaga.InferenceRequest(
            body.chatId(),
            userId,
            body.userMessageId(),
            body.text(),
            body.modelHint()
        );

        return saga.execute(req)
            .map(event -> ServerSentEvent.<InferenceSaga.TokenEvent>builder()
                .event(event.type())
                .data(event)
                .build())
            // Heartbeat â€” long-running stream'lerde proxy/load-balancer'Ä±n
            // baÄŸlantÄ±yÄ± koparmamasÄ± iÃ§in her 15s bir ping atÄ±lÄ±r.
            // (Åžu an basit, proper merge ile production'da sadece IDLE iken atÄ±lÄ±r)
            ;
    }

    private UUID userIdFromRequest(ServerHttpRequest request) {
        String raw = request.getHeaders().getFirst("X-User-Id");
        if (raw == null || raw.isBlank()) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Missing X-User-Id");
        }
        try {
            return UUID.fromString(raw);
        } catch (IllegalArgumentException ex) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Invalid X-User-Id");
        }
    }

    public record StreamRequest(
        @NotNull UUID chatId,
        @NotNull UUID userMessageId,
        @NotBlank String text,
        String modelHint
    ) {}
}


package com.goktug.inference.client;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import io.github.resilience4j.reactor.circuitbreaker.operator.CircuitBreakerOperator;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.MediaType;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.UUID;

/**
 * inference-worker-goktug client.
 *
 * Worker SSE endpoint'ine POST atar, dÃ¶nen SSE stream'i Flux<TokenEvent>
 * olarak yayÄ±nlar. Each event JSON'u {type, content, jobId, ...} formatÄ±nda.
 */
@Component
@Slf4j
public class InferenceWorkerClient {

    private final WebClient client;
    private final CircuitBreakerRegistry cbRegistry;
    private final ObjectMapper objectMapper;

    public InferenceWorkerClient(
            @Qualifier("workerGoktugWebClient") WebClient client,
            CircuitBreakerRegistry cbRegistry,
            ObjectMapper objectMapper
    ) {
        this.client = client;
        this.cbRegistry = cbRegistry;
        this.objectMapper = objectMapper;
    }

    public Flux<WorkerEvent> generateStream(GenerateRequest request) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("job_id", request.jobId().toString());
        body.put("prompt", request.prompt());
        body.put("max_tokens", request.maxTokens());
        body.put("temperature", request.temperature());
        body.put("top_k", request.topK());
        body.put("top_p", request.topP());
        body.put("repetition_penalty", request.repetitionPenalty());
        body.put("stream", true);

        ParameterizedTypeReference<ServerSentEvent<JsonNode>> typeRef =
            new ParameterizedTypeReference<>() {};

        return client.post()
            .uri("/v1/generate")
            .accept(MediaType.TEXT_EVENT_STREAM)
            .bodyValue(body)
            .retrieve()
            .bodyToFlux(typeRef)
            .mapNotNull(sse -> {
                JsonNode data = sse.data();
                if (data == null) return null;
                String type = data.has("type") ? data.get("type").asText() : "token";
                String content = data.has("content") ? data.get("content").asText() : "";
                Integer tokenCount = data.has("tokenCount") ? data.get("tokenCount").asInt() : null;
                Integer latencyMs = data.has("latencyMs") ? data.get("latencyMs").asInt() : null;
                String message = data.has("message") ? data.get("message").asText() : null;
                return new WorkerEvent(type, content, tokenCount, latencyMs, message);
            })
            .transformDeferred(CircuitBreakerOperator.of(cbRegistry.circuitBreaker("worker-goktug")));
    }

    public record GenerateRequest(
        UUID jobId,
        String prompt,
        int maxTokens,
        double temperature,
        int topK,
        double topP,
        double repetitionPenalty
    ) {}

    public record WorkerEvent(
        String type,           // "token" | "done" | "error"
        String content,
        Integer tokenCount,
        Integer latencyMs,
        String errorMessage
    ) {}
}


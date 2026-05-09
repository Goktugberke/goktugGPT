package com.goktug.inference.client;

import io.github.resilience4j.reactor.circuitbreaker.operator.CircuitBreakerOperator;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.Map;

/**
 * guardrail-service client. Faz 2'de gerÃ§ek model entegre edilince aktif olur.
 * Faz 1: dev profilinde bypass â€” herÅŸeyi safe kabul et.
 */
@Component
@Slf4j
public class GuardrailClient {

    private final WebClient client;
    private final CircuitBreakerRegistry cbRegistry;
    private final boolean enabled;

    public GuardrailClient(
            @Qualifier("guardrailClient") WebClient client,
            CircuitBreakerRegistry cbRegistry,
            @org.springframework.beans.factory.annotation.Value("${guardrail.enabled:false}") boolean enabled
    ) {
        this.client = client;
        this.cbRegistry = cbRegistry;
        this.enabled = enabled;
    }

    public Mono<GuardrailResult> check(String text) {
        if (!enabled) {
            log.debug("Guardrail disabled (Faz 1 dev) â€” allowing all");
            return Mono.just(new GuardrailResult(true, null));
        }

        return client.post()
            .uri("/v1/check")
            .bodyValue(Map.of("text", text))
            .retrieve()
            .bodyToMono(GuardrailResponse.class)
            .map(r -> new GuardrailResult(r.safe(), r.blocked_reason()))
            .transformDeferred(CircuitBreakerOperator.of(cbRegistry.circuitBreaker("guardrail")))
            .onErrorResume(ex -> {
                log.warn("Guardrail call failed, fail-open: {}", ex.getMessage());
                return Mono.just(new GuardrailResult(true, null));
            });
    }

    public record GuardrailResult(boolean safe, String reason) {}
    public record GuardrailResponse(boolean safe, Map<String, Double> categories, String blocked_reason) {}
}


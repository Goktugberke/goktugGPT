package com.goktug.inference.client;

import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import io.github.resilience4j.reactor.circuitbreaker.operator.CircuitBreakerOperator;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.Map;
import java.util.UUID;

/**
 * billing-service quota check. Faz 1'de bypass.
 */
@Component
@Slf4j
public class BillingClient {

    private final WebClient client;
    private final CircuitBreakerRegistry cbRegistry;
    private final boolean enabled;

    public BillingClient(
            @Qualifier("billingWebClient") WebClient client,
            CircuitBreakerRegistry cbRegistry,
            @Value("${billing.enabled:false}") boolean enabled
    ) {
        this.client = client;
        this.cbRegistry = cbRegistry;
        this.enabled = enabled;
    }

    public Mono<QuotaResult> checkQuota(UUID userId) {
        if (!enabled) {
            return Mono.just(new QuotaResult(true, Integer.MAX_VALUE));
        }
        return client.post()
            .uri("/internal/billing/check")
            .bodyValue(Map.of("userId", userId.toString()))
            .retrieve()
            .bodyToMono(QuotaResult.class)
            .transformDeferred(CircuitBreakerOperator.of(cbRegistry.circuitBreaker("billing")))
            .onErrorResume(ex -> {
                log.warn("Billing check failed, fail-open: {}", ex.getMessage());
                return Mono.just(new QuotaResult(true, 0));
            });
    }

    public Mono<Void> refund(UUID userId, UUID jobId, int tokens) {
        if (!enabled) return Mono.empty();
        return client.post()
            .uri("/internal/billing/refund")
            .bodyValue(Map.of(
                "userId", userId.toString(),
                "jobId", jobId.toString(),
                "tokens", tokens
            ))
            .retrieve()
            .bodyToMono(Void.class);
    }

    public record QuotaResult(boolean ok, int remaining) {}
}


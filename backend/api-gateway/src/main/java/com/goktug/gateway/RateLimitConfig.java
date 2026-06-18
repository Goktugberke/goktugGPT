package com.goktug.gateway;

import org.springframework.cloud.gateway.filter.ratelimit.KeyResolver;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import reactor.core.publisher.Mono;

/**
 * Rate limit anahtar çözümleyicisi.
 *
 * - JWT varsa: kullanıcı subject claim'ine göre rate limit (kullanıcı bazlı kota).
 * - JWT yoksa: IP bazlı (anonymous endpoint'ler için).
 *
 * Redis token bucket Spring Cloud Gateway'in built-in RedisRateLimiter'ı tarafından
 * uygulanır; biz sadece "key" üretiyoruz.
 */
@Configuration
public class RateLimitConfig {

    @Bean
    public KeyResolver userKeyResolver() {
        return exchange -> exchange.getPrincipal()
                .map(principal -> "user:" + principal.getName())
                .switchIfEmpty(Mono.fromSupplier(() -> {
                    String ip = exchange.getRequest()
                            .getRemoteAddress() != null
                            ? exchange.getRequest().getRemoteAddress().getAddress().getHostAddress()
                            : "unknown";
                    return "ip:" + ip;
                }));
    }
}

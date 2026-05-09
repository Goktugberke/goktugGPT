package com.goktug.gateway;

import org.springframework.cloud.gateway.filter.GatewayFilterChain;
import org.springframework.cloud.gateway.filter.GlobalFilter;
import org.springframework.core.Ordered;
import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.security.core.context.ReactiveSecurityContextHolder;
import org.springframework.security.core.context.SecurityContext;
import org.springframework.security.oauth2.jwt.Jwt;
import org.springframework.security.oauth2.server.resource.authentication.JwtAuthenticationToken;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Mono;

import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * JWT → downstream HTTP headers mapper.
 *
 * Spring Security ResourceServer JWT'yi validate eder; bu filter o JWT'den
 * kullanıcı bilgilerini çıkarıp downstream servislerin BEKLEDİĞİ header'lara
 * çevirir:
 *
 *   X-User-Id    ← jwt.sub
 *   X-User-Email ← jwt.email
 *   X-User-Roles ← jwt.realm_access.roles (virgülle ayrılmış)
 *
 * Böylece downstream servisler JWT signature validation tekrar yapmıyor
 * (gateway zaten yaptı) ve sadece basit header okuyorlar. mTLS Faz 3'te
 * gelecek; o zamana kadar internal network'te bu güvenli.
 *
 * NOT: Header'ları kötü niyetli client'in göndermesini engellemek için
 * gateway request'ten gelen aynı isimdeki header'ları SİLER ve kendisi yazar.
 */
@Component
public class JwtToHeadersFilter implements GlobalFilter, Ordered {

    public static final String USER_ID = "X-User-Id";
    public static final String USER_EMAIL = "X-User-Email";
    public static final String USER_ROLES = "X-User-Roles";

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        return ReactiveSecurityContextHolder.getContext()
            .map(SecurityContext::getAuthentication)
            .filter(auth -> auth instanceof JwtAuthenticationToken)
            .cast(JwtAuthenticationToken.class)
            .map(JwtAuthenticationToken::getToken)
            .flatMap(jwt -> chain.filter(propagateHeaders(exchange, jwt)))
            // No JWT (anonymous endpoint, e.g. /api/v1/auth/login) — strip
            // any incoming user headers so they can't be spoofed.
            .switchIfEmpty(chain.filter(stripHeaders(exchange)));
    }

    private ServerWebExchange propagateHeaders(ServerWebExchange exchange, Jwt jwt) {
        String sub = jwt.getSubject();
        String email = jwt.getClaimAsString("email");

        Map<String, Object> realmAccess = jwt.getClaimAsMap("realm_access");
        String rolesStr = "";
        if (realmAccess != null && realmAccess.get("roles") instanceof Collection<?> r) {
            rolesStr = String.join(",", r.stream().map(Object::toString).toList());
        }
        final String roles = rolesStr;

        ServerHttpRequest mutated = exchange.getRequest().mutate()
            .headers(h -> {
                h.remove(USER_ID);
                h.remove(USER_EMAIL);
                h.remove(USER_ROLES);
                if (sub != null) h.set(USER_ID, sub);
                if (email != null) h.set(USER_EMAIL, email);
                if (!roles.isEmpty()) h.set(USER_ROLES, roles);
            })
            .build();
        return exchange.mutate().request(mutated).build();
    }

    private ServerWebExchange stripHeaders(ServerWebExchange exchange) {
        ServerHttpRequest mutated = exchange.getRequest().mutate()
            .headers(h -> {
                h.remove(USER_ID);
                h.remove(USER_EMAIL);
                h.remove(USER_ROLES);
            })
            .build();
        return exchange.mutate().request(mutated).build();
    }

    @Override
    public int getOrder() {
        // Authentication filter'dan SONRA çalışmalı (security context dolu olsun)
        // Default Spring Security order = -100; biz -50'de çalışıyoruz.
        return -50;
    }
}

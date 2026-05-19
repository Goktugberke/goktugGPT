package com.goktug.inference.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.web.server.ServerHttpSecurity;
import org.springframework.security.web.server.SecurityWebFilterChain;

/**
 * Orchestrator sits behind the gateway. Authentication is already done there.
 * Disable CSRF (we accept POSTs from authenticated proxy traffic) and let all
 * requests through so SSE streams can be opened.
 */
@Configuration
public class SecurityConfig {

    @Bean
    public SecurityWebFilterChain securityWebFilterChain(ServerHttpSecurity http) {
        return http
            .csrf(csrf -> csrf.disable())
            .authorizeExchange(ex -> ex.anyExchange().permitAll())
            .build();
    }
}

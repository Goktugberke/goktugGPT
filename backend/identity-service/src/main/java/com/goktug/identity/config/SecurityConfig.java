package com.goktug.identity.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;

/**
 * Identity-service is reached only via the api-gateway, which already validates
 * incoming JWTs and strips spoofed headers. The auth endpoints (register/login/
 * refresh/logout) are public by design — they MUST be reachable without a
 * token. We therefore disable the default Spring Security chain that
 * `spring-boot-starter-oauth2-resource-server` brings in.
 *
 * Without this config, the default chain tried to JWT-protect every endpoint
 * and hung waiting for a JWKS issuer-uri that was never configured.
 */
@Configuration
public class SecurityConfig {

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        return http
            .csrf(csrf -> csrf.disable())
            .sessionManagement(s -> s.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
            .authorizeHttpRequests(auth -> auth.anyRequest().permitAll())
            .build();
    }
}

package com.goktug.identity.keycloak;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * Keycloak Admin REST API client.
 *
 * Bu servis Keycloak'ı yönetir:
 *   - Admin token al (client_credentials grant)
 *   - Yeni user oluştur (password set)
 *   - User'ı sil / disable et
 *   - User'ın token'ını al (password grant — login proxy)
 *
 * Production'da: Resilience4j circuit breaker + retry, token caching.
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class KeycloakClient {

    private final WebClient client;
    private final String realm;
    private final String adminClientId;
    private final String adminClientSecret;
    private final String publicClientId;

    public KeycloakClient(
            @Value("${keycloak.base-url}") String baseUrl,
            @Value("${keycloak.realm}") String realm,
            @Value("${keycloak.admin-client-id}") String adminClientId,
            @Value("${keycloak.admin-client-secret}") String adminClientSecret,
            @Value("${keycloak.public-client-id:goktuggpt-web}") String publicClientId
    ) {
        this.client = WebClient.builder().baseUrl(baseUrl).build();
        this.realm = realm;
        this.adminClientId = adminClientId;
        this.adminClientSecret = adminClientSecret;
        this.publicClientId = publicClientId;
    }

    public Mono<String> getAdminToken() {
        MultiValueMap<String, String> body = new LinkedMultiValueMap<>();
        body.add("grant_type", "client_credentials");
        body.add("client_id", adminClientId);
        body.add("client_secret", adminClientSecret);

        return client.post()
            .uri("/realms/{realm}/protocol/openid-connect/token", realm)
            .contentType(MediaType.APPLICATION_FORM_URLENCODED)
            .body(BodyInserters.fromFormData(body))
            .retrieve()
            .bodyToMono(TokenResponse.class)
            .map(TokenResponse::accessToken);
    }

    /**
     * Yeni Keycloak user oluştur — döner: user UUID (sub claim).
     */
    public Mono<UUID> createUser(String email, String password, String displayName) {
        return getAdminToken().flatMap(token -> {
            Map<String, Object> req = Map.of(
                "username", email,
                "email", email,
                "enabled", true,
                "emailVerified", true,
                "firstName", displayName == null ? "" : displayName,
                "credentials", List.of(Map.of(
                    "type", "password",
                    "value", password,
                    "temporary", false
                ))
            );
            return client.post()
                .uri("/admin/realms/{realm}/users", realm)
                .header("Authorization", "Bearer " + token)
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(req)
                .exchangeToMono(resp -> {
                    if (resp.statusCode().is2xxSuccessful()) {
                        // Keycloak Location header: /admin/realms/{realm}/users/{id}
                        String location = resp.headers().asHttpHeaders().getFirst("Location");
                        if (location == null) {
                            return Mono.error(new RuntimeException("Keycloak did not return Location header"));
                        }
                        String idStr = location.substring(location.lastIndexOf('/') + 1);
                        return Mono.just(UUID.fromString(idStr));
                    }
                    return resp.bodyToMono(String.class)
                        .defaultIfEmpty("(empty)")
                        .flatMap(b -> Mono.error(new RuntimeException(
                            "Keycloak createUser failed: " + resp.statusCode() + " " + b)));
                });
        });
    }

    /**
     * Email + password ile token al (login proxy).
     */
    public Mono<TokenResponse> loginWithPassword(String email, String password) {
        MultiValueMap<String, String> body = new LinkedMultiValueMap<>();
        body.add("grant_type", "password");
        body.add("client_id", publicClientId);
        body.add("username", email);
        body.add("password", password);

        return client.post()
            .uri("/realms/{realm}/protocol/openid-connect/token", realm)
            .contentType(MediaType.APPLICATION_FORM_URLENCODED)
            .body(BodyInserters.fromFormData(body))
            .retrieve()
            .bodyToMono(TokenResponse.class);
    }

    public Mono<TokenResponse> refresh(String refreshToken) {
        MultiValueMap<String, String> body = new LinkedMultiValueMap<>();
        body.add("grant_type", "refresh_token");
        body.add("client_id", publicClientId);
        body.add("refresh_token", refreshToken);

        return client.post()
            .uri("/realms/{realm}/protocol/openid-connect/token", realm)
            .contentType(MediaType.APPLICATION_FORM_URLENCODED)
            .body(BodyInserters.fromFormData(body))
            .retrieve()
            .bodyToMono(TokenResponse.class);
    }

    public Mono<Void> deleteUser(UUID userId) {
        return getAdminToken().flatMap(token ->
            client.delete()
                .uri("/admin/realms/{realm}/users/{id}", realm, userId)
                .header("Authorization", "Bearer " + token)
                .retrieve()
                .bodyToMono(Void.class));
    }

    public record TokenResponse(
        @JsonProperty("access_token") String accessToken,
        @JsonProperty("refresh_token") String refreshToken,
        @JsonProperty("expires_in") Integer expiresIn,
        @JsonProperty("refresh_expires_in") Integer refreshExpiresIn,
        @JsonProperty("token_type") String tokenType
    ) {}
}

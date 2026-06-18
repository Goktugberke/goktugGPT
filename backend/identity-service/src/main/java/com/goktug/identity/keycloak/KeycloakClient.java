package com.goktug.identity.keycloak;

import com.fasterxml.jackson.annotation.JsonProperty;
import io.netty.channel.ChannelOption;
import io.netty.handler.timeout.ReadTimeoutHandler;
import io.netty.handler.timeout.WriteTimeoutHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import reactor.netty.http.client.HttpClient;
import reactor.netty.resources.ConnectionProvider;

import java.time.Duration;
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
 */
@Component
public class KeycloakClient {

    private static final Logger log = LoggerFactory.getLogger(KeycloakClient.class);

    private final WebClient client;
    private final String realm;
    private final String adminClientId;
    private final String adminClientSecret;
    private final String publicClientId;

    // Keycloak's first JIT-compile + DB warmup hit (after a fresh container
    // boot) can take 15-20 seconds. Give it enough headroom.
    private static final Duration NETTY_TIMEOUT = Duration.ofSeconds(30);
    private static final Duration MONO_TIMEOUT = Duration.ofSeconds(25);

    public KeycloakClient(
            @Value("${keycloak.base-url}") String baseUrl,
            @Value("${keycloak.realm}") String realm,
            @Value("${keycloak.admin-client-id}") String adminClientId,
            @Value("${keycloak.admin-client-secret}") String adminClientSecret,
            @Value("${keycloak.public-client-id:goktuggpt-web}") String publicClientId
    ) {
        // Dedicated, small, fixed connection pool to avoid sharing pool with
        // anything else in the JVM (e.g. background exporter threads).
        ConnectionProvider pool = ConnectionProvider.builder("kc-pool")
            .maxConnections(20)
            .pendingAcquireTimeout(Duration.ofSeconds(5))
            .pendingAcquireMaxCount(50)
            .build();
        HttpClient http = HttpClient.create(pool)
            .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, 3000)
            .responseTimeout(NETTY_TIMEOUT)
            .doOnConnected(c -> c
                .addHandlerLast(new ReadTimeoutHandler((int) NETTY_TIMEOUT.toSeconds()))
                .addHandlerLast(new WriteTimeoutHandler(5)));
        this.client = WebClient.builder()
            .baseUrl(baseUrl)
            .clientConnector(new ReactorClientHttpConnector(http))
            .build();
        this.realm = realm;
        this.adminClientId = adminClientId;
        this.adminClientSecret = adminClientSecret;
        this.publicClientId = publicClientId;
        log.info("KeycloakClient initialised: baseUrl={} realm={} adminClientId={}",
            baseUrl, realm, adminClientId);
    }

    public Mono<String> getAdminToken() {
        log.info("KC: getAdminToken -> POST /realms/{}/protocol/openid-connect/token", realm);
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
            .map(TokenResponse::accessToken)
            .timeout(MONO_TIMEOUT)
            .doOnNext(t -> log.info("KC: getAdminToken OK (len={})", t == null ? 0 : t.length()))
            .doOnError(e -> log.warn("KC: getAdminToken FAILED: {}", e.toString()));
    }

    public Mono<UUID> createUser(String email, String password, String displayName) {
        log.info("KC: createUser email={}", email);
        return getAdminToken().flatMap(token -> {
            Map<String, Object> req = Map.of(
                "username", email,
                "email", email,
                "enabled", true,
                "emailVerified", true,
                "firstName", displayName == null ? "" : displayName,
                // Explicitly clear required actions so password grant works
                // immediately. Without this, Keycloak's default realm-level
                // required actions (verify-email, etc.) attach to the new user
                // and `password` grant returns "Account is not fully set up".
                "requiredActions", List.of(),
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
                    log.info("KC: createUser response status={}", resp.statusCode());
                    if (resp.statusCode().is2xxSuccessful()) {
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
        })
        .timeout(MONO_TIMEOUT)
        .doOnNext(id -> log.info("KC: createUser OK userId={}", id))
        .doOnError(e -> log.warn("KC: createUser FAILED: {}", e.toString()));
    }

    public Mono<TokenResponse> loginWithPassword(String email, String password) {
        log.info("KC: loginWithPassword email={}", email);
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
            .bodyToMono(TokenResponse.class)
            .timeout(MONO_TIMEOUT)
            .doOnNext(t -> log.info("KC: loginWithPassword OK"))
            .doOnError(e -> log.warn("KC: loginWithPassword FAILED: {}", e.toString()));
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
            .bodyToMono(TokenResponse.class)
            .timeout(MONO_TIMEOUT);
    }

    public Mono<Void> deleteUser(UUID userId) {
        return getAdminToken().flatMap(token ->
            client.delete()
                .uri("/admin/realms/{realm}/users/{id}", realm, userId)
                .header("Authorization", "Bearer " + token)
                .retrieve()
                .bodyToMono(Void.class))
            .timeout(MONO_TIMEOUT);
    }

    public record TokenResponse(
        @JsonProperty("access_token") String accessToken,
        @JsonProperty("refresh_token") String refreshToken,
        @JsonProperty("expires_in") Integer expiresIn,
        @JsonProperty("refresh_expires_in") Integer refreshExpiresIn,
        @JsonProperty("token_type") String tokenType
    ) {}
}

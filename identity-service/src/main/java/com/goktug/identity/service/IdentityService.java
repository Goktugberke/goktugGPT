package com.goktug.identity.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.goktug.identity.api.dto.AuthDtos.*;
import com.goktug.identity.domain.*;
import com.goktug.identity.error.IdentityException;
import com.goktug.identity.keycloak.KeycloakClient;
import com.goktug.identity.outbox.OutboxEntity;
import com.goktug.identity.outbox.OutboxRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Duration;
import java.time.OffsetDateTime;
import java.util.UUID;

@Service
@RequiredArgsConstructor
@Slf4j
public class IdentityService {

    private final ProfileRepository profileRepository;
    private final CustomInstructionsRepository customInstructionsRepository;
    private final OutboxRepository outboxRepository;
    private final KeycloakClient keycloak;
    private final ObjectMapper objectMapper;

    /**
     * Register flow:
     *   1. Email duplicate kontrolü
     *   2. Keycloak'ta user oluştur (id'yi al)
     *   3. profile DB row'u kaydet
     *   4. user.registered.v1 event'i Outbox'a yaz (aynı transaction)
     *
     * Saga aspect: Keycloak başarılı, ama DB başarısız → orphan Keycloak user
     * kalır. Production'da: önce DB'ye yazıp sonra Keycloak — fail durumunda
     * DB rollback. Veya saga state kullan (Faz 3).
     */
    @Transactional
    public TokenResponse register(RegisterRequest req) {
        log.info("register: START email={}", req.email());

        if (profileRepository.existsByEmail(req.email())) {
            log.info("register: email already registered");
            throw new IdentityException(HttpStatus.CONFLICT, "Email already registered");
        }
        log.info("register: email check passed");

        UUID userId;
        try {
            log.info("register: calling Keycloak createUser...");
            userId = keycloak.createUser(req.email(), req.password(), req.displayName())
                .block(Duration.ofSeconds(15));
            log.info("register: Keycloak createUser returned userId={}", userId);
        } catch (Exception ex) {
            log.error("register: Keycloak createUser failed: {}", ex.toString(), ex);
            throw new IdentityException(HttpStatus.BAD_REQUEST, "Could not create user: " + ex.getMessage());
        }
        if (userId == null) {
            throw new IdentityException(HttpStatus.INTERNAL_SERVER_ERROR, "Keycloak returned no user id");
        }

        log.info("register: saving profile");
        ProfileEntity profile = ProfileEntity.builder()
            .userId(userId)
            .email(req.email())
            .displayName(req.displayName())
            .build();
        profileRepository.save(profile);
        log.info("register: profile saved");

        // user.registered.v1 outbox event
        ObjectNode payload = objectMapper.createObjectNode();
        payload.put("eventId", UUID.randomUUID().toString());
        payload.put("eventType", "user.registered.v1");
        payload.put("occurredAt", OffsetDateTime.now().toString());
        payload.put("producer", "identity-service");
        payload.put("userId", userId.toString());

        ObjectNode innerPayload = objectMapper.createObjectNode();
        innerPayload.put("userId", userId.toString());
        innerPayload.put("email", req.email());
        innerPayload.put("displayName", req.displayName() == null ? "" : req.displayName());
        innerPayload.put("registeredAt", OffsetDateTime.now().toString());
        innerPayload.put("source", "self-service");
        payload.set("payload", innerPayload);

        log.info("register: writing outbox event");
        outboxRepository.save(OutboxEntity.builder()
            .aggregateId(userId)
            .eventType("user.registered.v1")
            .payload((JsonNode) payload)
            .build());
        log.info("register: outbox saved, auto-login starting");

        // Auto-login after register
        TokenResponse t = loginInternal(req.email(), req.password(), userId);
        log.info("register: DONE userId={}", userId);
        return t;
    }

    public TokenResponse login(LoginRequest req) {
        ProfileEntity profile = profileRepository.findByEmail(req.email())
            .orElseThrow(() -> new IdentityException(HttpStatus.UNAUTHORIZED, "Invalid credentials"));
        return loginInternal(req.email(), req.password(), profile.getUserId());
    }

    private TokenResponse loginInternal(String email, String password, UUID userId) {
        try {
            log.info("loginInternal: calling Keycloak loginWithPassword email={}", email);
            KeycloakClient.TokenResponse kc = keycloak.loginWithPassword(email, password)
                .block(Duration.ofSeconds(15));
            log.info("loginInternal: Keycloak login returned, token present={}", kc != null);
            if (kc == null) {
                throw new IdentityException(HttpStatus.UNAUTHORIZED, "Login failed");
            }
            return new TokenResponse(
                kc.accessToken(), kc.refreshToken(), kc.expiresIn(),
                userId.toString(), email);
        } catch (IdentityException e) { throw e; }
        catch (Exception ex) {
            log.warn("Keycloak login failed: {}", ex.toString(), ex);
            throw new IdentityException(HttpStatus.UNAUTHORIZED, "Invalid credentials");
        }
    }

    public TokenResponse refresh(String refreshToken) {
        try {
            KeycloakClient.TokenResponse kc = keycloak.refresh(refreshToken).block();
            if (kc == null) throw new IdentityException(HttpStatus.UNAUTHORIZED, "Refresh failed");
            return new TokenResponse(kc.accessToken(), kc.refreshToken(), kc.expiresIn(), null, null);
        } catch (IdentityException e) { throw e; }
        catch (Exception ex) {
            throw new IdentityException(HttpStatus.UNAUTHORIZED, "Invalid refresh token");
        }
    }

    @Transactional(readOnly = true)
    public ProfileDto getProfile(UUID userId) {
        ProfileEntity p = profileRepository.findById(userId)
            .orElseThrow(() -> new IdentityException(HttpStatus.NOT_FOUND, "Profile not found"));
        return toDto(p);
    }

    @Transactional
    public ProfileDto updateProfile(UUID userId, UpdateProfileRequest req) {
        ProfileEntity p = profileRepository.findById(userId)
            .orElseThrow(() -> new IdentityException(HttpStatus.NOT_FOUND, "Profile not found"));
        if (req.displayName() != null) p.setDisplayName(req.displayName());
        if (req.avatarUrl() != null) p.setAvatarUrl(req.avatarUrl());
        if (req.language() != null) p.setLanguage(req.language());
        if (req.theme() != null) p.setTheme(req.theme());
        return toDto(p);
    }

    @Transactional(readOnly = true)
    public CustomInstructionsDto getCustomInstructions(UUID userId) {
        return customInstructionsRepository.findById(userId)
            .map(c -> new CustomInstructionsDto(c.getAboutUser(), c.getResponseStyle(), c.getEnabled()))
            .orElse(new CustomInstructionsDto(null, null, true));
    }

    @Transactional
    public CustomInstructionsDto updateCustomInstructions(UUID userId, CustomInstructionsDto dto) {
        if (!profileRepository.existsById(userId)) {
            throw new IdentityException(HttpStatus.NOT_FOUND, "Profile not found");
        }
        CustomInstructionsEntity c = customInstructionsRepository.findById(userId)
            .orElseGet(() -> {
                CustomInstructionsEntity newEntity = new CustomInstructionsEntity();
                newEntity.setUserId(userId);
                return newEntity;
            });
        c.setAboutUser(dto.aboutUser());
        c.setResponseStyle(dto.responseStyle());
        c.setEnabled(dto.enabled() == null ? Boolean.TRUE : dto.enabled());
        customInstructionsRepository.save(c);
        return new CustomInstructionsDto(c.getAboutUser(), c.getResponseStyle(), c.getEnabled());
    }

    private ProfileDto toDto(ProfileEntity p) {
        return new ProfileDto(
            p.getUserId().toString(), p.getEmail(),
            p.getDisplayName(), p.getAvatarUrl(),
            p.getLanguage(), p.getTheme()
        );
    }
}

package com.goktug.conversation.idempotency;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.OffsetDateTime;
import java.util.Optional;
import java.util.UUID;
import java.util.function.Supplier;

/**
 * Idempotency service.
 *
 * Pattern: cache-aside.
 *   1. (userId, key, endpoint) lookup et
 *   2. Cache hit varsa eski response'u dön
 *   3. Yoksa supplier'ı çalıştır, response'u kaydet, dön
 *
 * NOT: Bu basit (eventually consistent) implementasyon. Race condition
 * (aynı anda 2 request) durumunda DB'nin unique constraint'i devreye girer
 * → ikincisi exception alır → re-fetch + return.
 *
 * Production'da daha agresif: SELECT ... FOR UPDATE + INSERT IF NOT EXISTS
 * veya Redis NX SET ile lock.
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class IdempotencyService {

    private final IdempotencyRepository repository;
    private final ObjectMapper objectMapper;

    /**
     * Cached result fetch — yoksa boş Optional döner.
     */
    public Optional<CachedResponse> findCached(UUID userId, String key, String endpoint) {
        return repository.findByUserIdAndIdempotencyKeyAndEndpoint(userId, key, endpoint)
            .filter(e -> e.getExpiresAt().isAfter(OffsetDateTime.now()))
            .map(e -> new CachedResponse(e.getResponseStatus(), e.getResponseBody()));
    }

    /**
     * Yeni bir idempotent execution kaydet.
     */
    @Transactional
    public <T> T executeIdempotent(
            UUID userId,
            String key,
            String endpoint,
            Supplier<ResultWithStatus<T>> action
    ) {
        // Önce cache'e bak
        Optional<IdempotencyEntity> cached = repository
            .findByUserIdAndIdempotencyKeyAndEndpoint(userId, key, endpoint)
            .filter(e -> e.getExpiresAt().isAfter(OffsetDateTime.now()));

        if (cached.isPresent()) {
            log.debug("Idempotency hit: user={} key={} endpoint={}", userId, key, endpoint);
            try {
                @SuppressWarnings("unchecked")
                T result = (T) objectMapper.treeToValue(
                    cached.get().getResponseBody(), Object.class);
                return result;
            } catch (Exception ex) {
                log.warn("Failed to deserialize cached idempotent response, will re-execute", ex);
            }
        }

        // İşi yap
        ResultWithStatus<T> result = action.get();

        // Kaydet
        try {
            JsonNode body = objectMapper.valueToTree(result.body());
            IdempotencyEntity entity = IdempotencyEntity.builder()
                .userId(userId)
                .idempotencyKey(key)
                .endpoint(endpoint)
                .responseStatus(result.status())
                .responseBody(body)
                .build();
            repository.save(entity);
        } catch (org.springframework.dao.DataIntegrityViolationException dup) {
            // Race condition: başka thread bu key'i kaydetmiş — yine ok
            log.debug("Idempotency race detected for key={}, treating as no-op", key);
        }

        return result.body();
    }

    @Scheduled(fixedDelayString = "${idempotency.cleanup-interval-ms:3600000}")  // 1 saat
    @Transactional
    public void cleanupExpired() {
        int n = repository.deleteExpired(OffsetDateTime.now());
        if (n > 0) log.info("Idempotency cleanup: removed {} expired entries", n);
    }

    public record CachedResponse(int status, JsonNode body) {}
    public record ResultWithStatus<T>(int status, T body) {}
}

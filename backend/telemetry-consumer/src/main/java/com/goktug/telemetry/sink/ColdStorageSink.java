package com.goktug.telemetry.sink;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.goktug.telemetry.config.TelemetryProperties;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.S3Configuration;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;
import software.amazon.awssdk.services.s3.model.CreateBucketRequest;
import software.amazon.awssdk.services.s3.model.HeadBucketRequest;
import software.amazon.awssdk.services.s3.model.NoSuchBucketException;

import java.net.URI;
import java.time.OffsetDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Buffers events in memory and flushes them as line-delimited JSON (NDJSON)
 * objects to S3-compatible cold storage on either a size or time trigger.
 *
 * Object key layout: {category}/{yyyy/MM/dd/HH}/{uuid}.ndjson — partitioned
 * so a downstream RLHF or analytics job can scan a date range cheaply.
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class ColdStorageSink {

    private static final DateTimeFormatter PATH_FMT = DateTimeFormatter.ofPattern("yyyy/MM/dd/HH");

    private final TelemetryProperties props;
    private final ObjectMapper objectMapper;

    private S3Client s3;
    private final ConcurrentLinkedQueue<BufferedEvent> buffer = new ConcurrentLinkedQueue<>();

    private record BufferedEvent(String category, JsonNode envelope) {}

    @PostConstruct
    void init() {
        if (!props.getColdStorage().isEnabled()) {
            log.info("Cold storage disabled");
            return;
        }
        TelemetryProperties.ColdStorage cfg = props.getColdStorage();
        s3 = S3Client.builder()
            .endpointOverride(URI.create(cfg.getEndpoint()))
            .region(Region.of(cfg.getRegion()))
            .credentialsProvider(StaticCredentialsProvider.create(
                AwsBasicCredentials.create(cfg.getAccessKey(), cfg.getSecretKey())))
            .serviceConfiguration(S3Configuration.builder().pathStyleAccessEnabled(true).build())
            .build();
        ensureBucket(cfg.getBucket());
        log.info("Cold storage ready: bucket={} endpoint={}", cfg.getBucket(), cfg.getEndpoint());
    }

    @PreDestroy
    void shutdown() {
        flush();
        if (s3 != null) s3.close();
    }

    public void append(String category, JsonNode envelope) {
        if (!props.getColdStorage().isEnabled() || s3 == null) return;
        buffer.add(new BufferedEvent(category, envelope));
        if (buffer.size() >= props.getColdStorage().getBatchSize()) {
            flush();
        }
    }

    @Scheduled(fixedDelayString = "${telemetry.cold-storage.flush-interval-ms:30000}")
    public void flushScheduled() {
        flush();
    }

    private synchronized void flush() {
        if (!props.getColdStorage().isEnabled() || s3 == null || buffer.isEmpty()) return;

        List<BufferedEvent> batch = new ArrayList<>(buffer.size());
        BufferedEvent ev;
        while ((ev = buffer.poll()) != null) batch.add(ev);
        if (batch.isEmpty()) return;

        // Partition by category (one object per category per flush).
        java.util.Map<String, StringBuilder> byCategory = new java.util.HashMap<>();
        for (BufferedEvent e : batch) {
            StringBuilder sb = byCategory.computeIfAbsent(e.category(), k -> new StringBuilder());
            try {
                sb.append(objectMapper.writeValueAsString(e.envelope())).append('\n');
            } catch (Exception ex) {
                log.debug("Skip event during serialize: {}", ex.getMessage());
            }
        }

        String path = OffsetDateTime.now().format(PATH_FMT);
        for (var entry : byCategory.entrySet()) {
            String key = "%s/%s/%s.ndjson".formatted(entry.getKey(), path, UUID.randomUUID());
            try {
                byte[] body = entry.getValue().toString().getBytes(java.nio.charset.StandardCharsets.UTF_8);
                s3.putObject(
                    PutObjectRequest.builder()
                        .bucket(props.getColdStorage().getBucket())
                        .key(key)
                        .contentType("application/x-ndjson")
                        .build(),
                    RequestBody.fromBytes(body)
                );
                log.info("Cold flushed key={} bytes={}", key, body.length);
            } catch (Exception ex) {
                log.warn("Cold flush failed key={}: {}", key, ex.getMessage());
            }
        }
    }

    private void ensureBucket(String bucket) {
        try {
            s3.headBucket(HeadBucketRequest.builder().bucket(bucket).build());
        } catch (NoSuchBucketException e) {
            try {
                s3.createBucket(CreateBucketRequest.builder().bucket(bucket).build());
                log.info("Created cold storage bucket: {}", bucket);
            } catch (Exception ex) {
                log.warn("Cold storage bucket create failed (will retry on write): {}", ex.getMessage());
            }
        } catch (Exception e) {
            log.warn("Cold storage bucket head failed: {}", e.getMessage());
        }
    }
}

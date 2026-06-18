package com.goktug.asset.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.goktug.asset.api.dto.AssetDtos.*;
import com.goktug.asset.domain.AssetEntity;
import com.goktug.asset.domain.AssetRepository;
import com.goktug.asset.error.AssetException;
import com.goktug.asset.storage.PresignedUrlService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import software.amazon.awssdk.services.s3.model.HeadObjectResponse;
import software.amazon.awssdk.services.s3.model.NoSuchKeyException;

import java.time.OffsetDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.UUID;

@Service
@RequiredArgsConstructor
@Slf4j
public class AssetService {

    private final AssetRepository repository;
    private final PresignedUrlService presigner;
    private final KafkaTemplate<String, String> kafkaTemplate;
    private final ObjectMapper objectMapper;

    @Value("${asset.max-size-bytes}")
    private long maxSizeBytes;

    @Value("${asset.allowed-mime-types}")
    private List<String> allowedMimeTypes;

    @Value("${minio.presigned-url-ttl-min}")
    private int ttlMinutes;

    @Transactional
    public UploadUrlResponse createUploadUrl(UUID userId, UploadUrlRequest req) {
        if (req.sizeBytes() > maxSizeBytes) {
            throw new AssetException(HttpStatus.BAD_REQUEST,
                "File too large; max " + maxSizeBytes + " bytes");
        }
        if (!allowedMimeTypes.contains(req.mimeType())) {
            throw new AssetException(HttpStatus.BAD_REQUEST,
                "MIME type not allowed: " + req.mimeType());
        }

        UUID assetId = UUID.randomUUID();
        String storagePath = buildStoragePath(userId, assetId, req.filename());

        AssetEntity entity = AssetEntity.builder()
            .id(assetId)
            .userId(userId)
            .chatId(req.chatId())
            .originalName(req.filename())
            .mimeType(req.mimeType())
            .sizeBytes(req.sizeBytes())
            .storagePath(storagePath)
            .status(AssetEntity.Status.PENDING_UPLOAD)
            .build();
        repository.save(entity);

        String url = presigner.generatePutUrl(storagePath, req.mimeType());
        return new UploadUrlResponse(
            assetId, url, storagePath,
            OffsetDateTime.now().plusMinutes(ttlMinutes));
    }

    @Transactional
    public AssetDto confirm(UUID userId, UUID assetId) {
        AssetEntity asset = repository.findByIdAndUserIdAndDeletedAtIsNull(assetId, userId)
            .orElseThrow(() -> new AssetException(HttpStatus.NOT_FOUND, "Asset not found"));

        if (asset.getStatus() == AssetEntity.Status.READY) {
            // Idempotent — zaten confirm edilmiş
            return toDto(asset);
        }

        // MinIO'ya HEAD at: gerçekten upload edildi mi?
        HeadObjectResponse head;
        try {
            head = presigner.head(asset.getStoragePath());
        } catch (NoSuchKeyException ex) {
            throw new AssetException(HttpStatus.BAD_REQUEST,
                "Object not uploaded yet at " + asset.getStoragePath());
        } catch (Exception ex) {
            log.error("S3 head failed", ex);
            throw new AssetException(HttpStatus.INTERNAL_SERVER_ERROR, "Storage check failed");
        }

        if (head.contentLength() != asset.getSizeBytes()) {
            log.warn("Size mismatch: declared={} actual={}", asset.getSizeBytes(), head.contentLength());
            asset.setSizeBytes(head.contentLength());   // gerçek boyutu kabul et
        }

        asset.setStatus(AssetEntity.Status.READY);
        asset.setConfirmedAt(OffsetDateTime.now());
        if (head.eTag() != null) {
            asset.setChecksumSha256(head.eTag().replaceAll("\"", ""));
        }
        repository.save(asset);

        publishUploadedEvent(asset);
        return toDto(asset);
    }

    @Transactional(readOnly = true)
    public DownloadUrlResponse getDownloadUrl(UUID userId, UUID assetId) {
        AssetEntity asset = repository.findByIdAndUserIdAndDeletedAtIsNull(assetId, userId)
            .orElseThrow(() -> new AssetException(HttpStatus.NOT_FOUND, "Asset not found"));
        if (asset.getStatus() != AssetEntity.Status.READY) {
            throw new AssetException(HttpStatus.CONFLICT, "Asset not ready");
        }
        String url = presigner.generateGetUrl(asset.getStoragePath());
        return new DownloadUrlResponse(url, OffsetDateTime.now().plusMinutes(ttlMinutes));
    }

    @Transactional(readOnly = true)
    public AssetDto get(UUID userId, UUID assetId) {
        AssetEntity asset = repository.findByIdAndUserIdAndDeletedAtIsNull(assetId, userId)
            .orElseThrow(() -> new AssetException(HttpStatus.NOT_FOUND, "Asset not found"));
        return toDto(asset);
    }

    @Transactional
    public void delete(UUID userId, UUID assetId) {
        AssetEntity asset = repository.findByIdAndUserIdAndDeletedAtIsNull(assetId, userId)
            .orElseThrow(() -> new AssetException(HttpStatus.NOT_FOUND, "Asset not found"));
        asset.setDeletedAt(OffsetDateTime.now());
        asset.setStatus(AssetEntity.Status.DELETED);
        repository.save(asset);
        // TODO: blob'u da sil — async job (publish asset.deleted.v1)
    }

    private void publishUploadedEvent(AssetEntity asset) {
        ObjectNode envelope = objectMapper.createObjectNode();
        envelope.put("eventId", UUID.randomUUID().toString());
        envelope.put("eventType", "asset.uploaded.v1");
        envelope.put("occurredAt", OffsetDateTime.now().toString());
        envelope.put("producer", "asset-service");
        envelope.put("userId", asset.getUserId().toString());

        ObjectNode payload = objectMapper.createObjectNode();
        payload.put("assetId", asset.getId().toString());
        payload.put("userId", asset.getUserId().toString());
        if (asset.getChatId() != null) payload.put("chatId", asset.getChatId().toString());
        payload.put("filename", asset.getOriginalName());
        payload.put("mimeType", asset.getMimeType());
        payload.put("sizeBytes", asset.getSizeBytes());
        payload.put("storagePath", asset.getStoragePath());
        if (asset.getChecksumSha256() != null) payload.put("checksumSha256", asset.getChecksumSha256());
        payload.put("uploadedAt", OffsetDateTime.now().toString());
        envelope.set("payload", payload);

        kafkaTemplate.send("asset.events", asset.getId().toString(), envelope.toString());
    }

    private String buildStoragePath(UUID userId, UUID assetId, String filename) {
        // uploads/{userId}/{yyyy}/{MM}/{dd}/{assetId}/{safeName}
        String date = OffsetDateTime.now().format(DateTimeFormatter.ofPattern("yyyy/MM/dd"));
        String safe = filename.replaceAll("[^a-zA-Z0-9._-]", "_");
        return String.format("uploads/%s/%s/%s/%s", userId, date, assetId, safe);
    }

    private AssetDto toDto(AssetEntity a) {
        return new AssetDto(
            a.getId(), a.getUserId(), a.getChatId(),
            a.getOriginalName(), a.getMimeType(), a.getSizeBytes(),
            a.getStatus().name(), a.getCreatedAt());
    }
}

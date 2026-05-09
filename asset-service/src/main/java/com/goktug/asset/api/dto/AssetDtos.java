package com.goktug.asset.api.dto;

import jakarta.validation.constraints.*;

import java.time.OffsetDateTime;
import java.util.UUID;

public class AssetDtos {

    public record UploadUrlRequest(
        @NotBlank @Size(max = 512) String filename,
        @NotBlank @Size(max = 120) String mimeType,
        @Min(1) @Max(52428800) long sizeBytes,
        UUID chatId
    ) {}

    public record UploadUrlResponse(
        UUID assetId,
        String uploadUrl,
        String storagePath,
        OffsetDateTime expiresAt
    ) {}

    public record AssetDto(
        UUID id,
        UUID userId,
        UUID chatId,
        String originalName,
        String mimeType,
        Long sizeBytes,
        String status,
        OffsetDateTime createdAt
    ) {}

    public record DownloadUrlResponse(String downloadUrl, OffsetDateTime expiresAt) {}
}

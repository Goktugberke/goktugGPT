package com.goktug.asset.api;

import com.goktug.asset.api.dto.AssetDtos.*;
import com.goktug.asset.error.AssetException;
import com.goktug.asset.service.AssetService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.UUID;

@RestController
@RequestMapping("/api/v1/assets")
@RequiredArgsConstructor
public class AssetController {

    private final AssetService assetService;

    @PostMapping("/upload-url")
    public ResponseEntity<UploadUrlResponse> uploadUrl(
            @RequestBody @Valid UploadUrlRequest req,
            HttpServletRequest request) {
        UUID userId = userIdOrThrow(request);
        return ResponseEntity.status(HttpStatus.CREATED).body(assetService.createUploadUrl(userId, req));
    }

    @PostMapping("/{assetId}/confirm")
    public AssetDto confirm(@PathVariable UUID assetId, HttpServletRequest request) {
        UUID userId = userIdOrThrow(request);
        return assetService.confirm(userId, assetId);
    }

    @GetMapping("/{assetId}")
    public AssetDto get(@PathVariable UUID assetId, HttpServletRequest request) {
        return assetService.get(userIdOrThrow(request), assetId);
    }

    @GetMapping("/{assetId}/download-url")
    public DownloadUrlResponse downloadUrl(@PathVariable UUID assetId, HttpServletRequest request) {
        return assetService.getDownloadUrl(userIdOrThrow(request), assetId);
    }

    @DeleteMapping("/{assetId}")
    public ResponseEntity<Void> delete(@PathVariable UUID assetId, HttpServletRequest request) {
        assetService.delete(userIdOrThrow(request), assetId);
        return ResponseEntity.noContent().build();
    }

    private UUID userIdOrThrow(HttpServletRequest request) {
        String header = request.getHeader("X-User-Id");
        if (header == null || header.isBlank()) {
            throw new AssetException(HttpStatus.UNAUTHORIZED, "Missing X-User-Id");
        }
        try { return UUID.fromString(header); }
        catch (IllegalArgumentException ex) {
            throw new AssetException(HttpStatus.UNAUTHORIZED, "Invalid X-User-Id");
        }
    }
}

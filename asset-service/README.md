# asset-service

> **Dosya yönetimi.** Saritaygpt'taki `AzureDataLakeService`'in mikroservis hali — MinIO/S3 backend.

## Sorumluluklar

- File metadata CRUD (DB)
- Presigned upload/download URL üretme (MinIO/S3)
- File validation (Tika ile MIME doğrulama, allowed types whitelist, size limit)
- `asset.uploaded.v1` event publish (Outbox)
- Cleanup: `user.deleted.v1` consume → kullanıcının tüm asset'lerini sil

## Mimari Karar — Presigned URL

Saritaygpt'ta dosya `MultipartFile` olarak backend'e geliyor → backend Azure'a forward ediyor → backend ağ I/O için bekliyor.

**Yeni:** Frontend direkt MinIO'ya basıyor (presigned PUT URL).

```
Frontend                      asset-service              MinIO
   │                                │                       │
   │  POST /assets/upload-url       │                       │
   │  ────────────────────────────▶ │                       │
   │  {assetId, presignedUrl}       │                       │
   │  ◀──────────────────────────── │                       │
   │                                                        │
   │  PUT presignedUrl  (binary)                            │
   │  ──────────────────────────────────────────────────▶   │
   │  200 OK                                                │
   │  ◀──────────────────────────────────────────────────   │
   │                                                        │
   │  POST /assets/{id}/confirm     │                       │
   │  ────────────────────────────▶ │ (validate Tika)       │
   │                                │ ────────────────▶     │ HEAD
   │                                │ ◀────────────────     │
   │                                │ status = READY        │
   │                                │ outbox: asset.uploaded│
```

→ Backend bandwidth tasarrufu, scalable upload, multi-part upload mümkün.

## Endpoints

| Method | Path | Açıklama |
|--------|------|----------|
| POST | /api/v1/assets/upload-url | Presigned PUT URL döner |
| POST | /api/v1/assets/{assetId}/confirm | Upload tamamlandı, validate et |
| GET | /api/v1/assets/{assetId}/download-url | Presigned GET URL |
| GET | /api/v1/assets/{assetId} | Metadata |
| DELETE | /api/v1/assets/{assetId} | Soft delete |

## Eventler

**Publish:**
- `asset.uploaded.v1` — multimodal-pipeline (Faz 2 — OCR/STT) dinler
- `asset.deleted.v1`

**Subscribe:**
- `user.deleted.v1` → o user'ın tüm asset'lerini soft-delete

## Patterns

- **Transactional Outbox** (asset.uploaded event)
- **Idempotency** (confirm endpoint için — duplicate validation çağrısı no-op)
- **Two-phase commit (logical):** PENDING_UPLOAD → READY (confirm sonrası)

## Port

`8084` (internal)

## TODO (bir sonraki session)

1. **JPA entity'leri:** `AssetEntity`, `AssetRepository`, `OutboxEntity` (conversation-service'ten kopyala)
2. **`MinioConfig`:** `S3Client` bean (AWS SDK v2)
3. **`PresignedUrlService`:** PUT/GET URL üretici
4. **`AssetController`:**
   - `POST /upload-url` → asset row oluştur (status=PENDING_UPLOAD), presigned URL üret
   - `POST /{id}/confirm` → MinIO'ya HEAD at (boyut + ETag check), Tika ile MIME doğrula, status=READY, outbox event
5. **`MimeValidator`:** Tika ile actual content-type tespit, whitelist'e uyuyor mu?
6. **`UserDeletedConsumer`:** `@KafkaListener("user.events")` → cascade delete
7. **OutboxPoller** (conversation-service'ten kopyala)
8. **CDN config (Faz 3):** CloudFront / CloudFlare ile presigned URL'leri cache'le

## Çalıştırma

```bash
docker compose --profile infra up -d
mvn -pl services/asset-service -am spring-boot:run
```

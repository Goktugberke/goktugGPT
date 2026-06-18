package com.goktug.asset.storage;

import io.github.resilience4j.bulkhead.annotation.Bulkhead;
import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import io.github.resilience4j.retry.annotation.Retry;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.HeadObjectRequest;
import software.amazon.awssdk.services.s3.model.HeadObjectResponse;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;
import software.amazon.awssdk.services.s3.presigner.S3Presigner;
import software.amazon.awssdk.services.s3.presigner.model.GetObjectPresignRequest;
import software.amazon.awssdk.services.s3.presigner.model.PutObjectPresignRequest;

import java.time.Duration;

@Service
@RequiredArgsConstructor
public class PresignedUrlService {

    private final S3Client s3;
    private final S3Presigner presigner;

    @Value("${minio.bucket}")
    private String bucket;

    @Value("${minio.presigned-url-ttl-min}")
    private int ttlMinutes;

    public String generatePutUrl(String key, String contentType) {
        PutObjectRequest put = PutObjectRequest.builder()
            .bucket(bucket).key(key).contentType(contentType).build();
        PutObjectPresignRequest req = PutObjectPresignRequest.builder()
            .signatureDuration(Duration.ofMinutes(ttlMinutes))
            .putObjectRequest(put).build();
        return presigner.presignPutObject(req).url().toString();
    }

    public String generateGetUrl(String key) {
        GetObjectRequest get = GetObjectRequest.builder().bucket(bucket).key(key).build();
        GetObjectPresignRequest req = GetObjectPresignRequest.builder()
            .signatureDuration(Duration.ofMinutes(ttlMinutes))
            .getObjectRequest(get).build();
        return presigner.presignGetObject(req).url().toString();
    }

    /**
     * MinIO'ya HEAD at — obje gerçekten yüklendi mi, kaç byte, hash ne?
     *
     * Resilience4j ile sarılı: MinIO çökerse hızlı fail, transient hata
     * durumunda 3 retry + exponential backoff. Bulkhead aşırı paralel
     * confirm çağrılarını queue'lar (50 concurrent max).
     */
    @CircuitBreaker(name = "minio")
    @Retry(name = "minio")
    @Bulkhead(name = "minio")
    public HeadObjectResponse head(String key) {
        return s3.headObject(HeadObjectRequest.builder().bucket(bucket).key(key).build());
    }
}

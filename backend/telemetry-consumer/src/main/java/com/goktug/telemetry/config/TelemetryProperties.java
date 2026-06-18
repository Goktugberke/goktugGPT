package com.goktug.telemetry.config;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Configuration
@ConfigurationProperties(prefix = "telemetry")
@Getter
@Setter
public class TelemetryProperties {

    private Es es = new Es();
    private ColdStorage coldStorage = new ColdStorage();

    @Getter @Setter
    public static class Es {
        private boolean enabled = true;
        private String indexPrefix = "telemetry";
    }

    @Getter @Setter
    public static class ColdStorage {
        private boolean enabled = true;
        private String endpoint = "http://minio:9000";
        private String region = "us-east-1";
        private String bucket = "goktug-telemetry";
        private String accessKey = "minio";
        private String secretKey = "minio12345";
        private int batchSize = 500;
        private long flushIntervalMs = 30_000;
    }
}

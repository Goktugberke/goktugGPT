package com.goktug.billing.config;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Configuration
@ConfigurationProperties(prefix = "billing.free-plan")
@Getter
@Setter
public class FreePlanProperties {
    private String name = "free";
    private long monthlyTokenQuota = 100_000;
    private int rateLimitCapacity = 1000;
    private double rateLimitRefillPerSec = 10.0;
}

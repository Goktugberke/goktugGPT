package com.goktug.notification.config;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Configuration
@ConfigurationProperties(prefix = "notification.mail")
@Getter
@Setter
public class MailProperties {
    private String from = "no-reply@goktug.dev";
    private boolean enabled = false;
}

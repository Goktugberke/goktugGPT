package com.goktug.telemetry.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestClient;

@Configuration
public class HttpClientConfig {

    @Bean
    public RestClient esRestClient(@Value("${spring.elasticsearch.uris}") String esUri) {
        return RestClient.builder().baseUrl(esUri).build();
    }
}

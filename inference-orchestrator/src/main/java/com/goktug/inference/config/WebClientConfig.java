package com.goktug.inference.config;

import io.netty.channel.ChannelOption;
import io.netty.handler.timeout.ReadTimeoutHandler;
import io.netty.handler.timeout.WriteTimeoutHandler;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.netty.http.client.HttpClient;

import java.time.Duration;

/**
 * WebClient bean factory for downstream service calls.
 *
 * Her downstream için ayrı WebClient bean'i — base URL ve timeout
 * konfigürasyonu farklı olabilir (guardrail = 2s, worker = 60s gibi).
 */
@Configuration
public class WebClientConfig {

    @Value("${downstream.guardrail}")
    private String guardrailBaseUrl;

    @Value("${downstream.billing}")
    private String billingBaseUrl;

    @Value("${downstream.router}")
    private String routerBaseUrl;

    @Value("${downstream.worker-goktug}")
    private String workerGoktugBaseUrl;

    @Bean(name = "guardrailWebClient")
    public WebClient guardrailClient() {
        return webClient(guardrailBaseUrl, Duration.ofSeconds(3));
    }

    @Bean(name = "billingWebClient")
    public WebClient billingClient() {
        return webClient(billingBaseUrl, Duration.ofSeconds(2));
    }

    @Bean(name = "routerWebClient")
    public WebClient routerClient() {
        return webClient(routerBaseUrl, Duration.ofSeconds(2));
    }

    @Bean(name = "workerGoktugWebClient")
    public WebClient workerGoktugClient() {
        // Streaming için uzun timeout, hangup'a izin ver
        return webClient(workerGoktugBaseUrl, Duration.ofSeconds(120));
    }

    private WebClient webClient(String baseUrl, Duration readTimeout) {
        HttpClient httpClient = HttpClient.create()
            .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, 3000)
            .responseTimeout(readTimeout)
            .doOnConnected(conn ->
                conn.addHandlerLast(new ReadTimeoutHandler((int) readTimeout.toSeconds()))
                    .addHandlerLast(new WriteTimeoutHandler(10)));

        return WebClient.builder()
            .baseUrl(baseUrl)
            .clientConnector(new ReactorClientHttpConnector(httpClient))
            .build();
    }
}

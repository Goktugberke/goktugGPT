package com.goktug.inference.client;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.Map;

@Component
@Slf4j
public class RouterClient {

    private final WebClient webClient;

    public RouterClient(@Qualifier("routerWebClient") WebClient webClient) {
        this.webClient = webClient;
    }

    public Mono<RouteResponse> pickModel(String prompt) {
        return webClient.post()
            .uri("/v1/route")
            .bodyValue(Map.of("prompt", prompt))
            .retrieve()
            .bodyToMono(RouteResponse.class)
            .onErrorResume(ex -> {
                log.warn("Router call failed, defaulting to goktug-mini: {}", ex.getMessage());
                return Mono.just(new RouteResponse("goktug-mini", 0.0));
            });
    }

    public record RouteResponse(String model, double confidence) {}
}

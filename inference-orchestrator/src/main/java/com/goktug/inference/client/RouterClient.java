package com.goktug.inference.client;

import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.Map;

@Component
@RequiredArgsConstructor
public class RouterClient {

    private final WebClient webClient;

    

    public Mono<RouteResponse> pickModel(String prompt) {
        return webClient.post()
            .uri("/v1/route")
            .bodyValue(Map.of("prompt", prompt))
            .retrieve()
            .bodyToMono(RouteResponse.class)
            .onErrorResume(ex -> Mono.just(new RouteResponse("goktug-mini", 0.0)));
    }

    public record RouteResponse(String model, double confidence) {}
}

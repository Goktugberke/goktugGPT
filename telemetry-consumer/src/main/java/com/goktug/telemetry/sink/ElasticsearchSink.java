package com.goktug.telemetry.sink;

import com.fasterxml.jackson.databind.JsonNode;
import com.goktug.telemetry.config.TelemetryProperties;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;

/**
 * Forwards each event to a date-suffixed Elasticsearch index, e.g.
 * `telemetry-inference-2026-05-13`. ES handles auto-creation.
 *
 * Uses RestClient directly (not spring-data-elasticsearch repositories) because:
 *   - We're indexing JsonNode envelopes with no fixed schema.
 *   - Spring Data ES requires entity classes per index.
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class ElasticsearchSink {

    private static final DateTimeFormatter DATE_FMT = DateTimeFormatter.ISO_LOCAL_DATE;

    private final RestClient esRestClient;
    private final TelemetryProperties props;

    public void index(String category, String eventType, JsonNode envelope) {
        if (!props.getEs().isEnabled()) return;

        String index = "%s-%s-%s".formatted(
            props.getEs().getIndexPrefix(),
            category,
            LocalDate.now().format(DATE_FMT)
        );

        try {
            esRestClient.post()
                .uri("/{index}/_doc", index)
                .contentType(MediaType.APPLICATION_JSON)
                .body(envelope)
                .retrieve()
                .toBodilessEntity();
            log.debug("Indexed event {} to {}", eventType, index);
        } catch (Exception e) {
            log.warn("ES index failed (index={}, eventType={}): {}", index, eventType, e.getMessage());
        }
    }
}

package com.goktug.telemetry.event;

import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

@Component
@Slf4j
public class TelemetryConsumer {

    

    @KafkaListener(topics = "telemetry.events", groupId = "telemetry-group")
    public void consume(String message) {
        log.info("Received telemetry event: {}", message);
        // Here you would parse the message and save it to a time-series database
    }
}

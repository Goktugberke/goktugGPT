package com.goktug.identity.outbox;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.PageRequest;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.time.OffsetDateTime;
import java.util.List;

@Component
@RequiredArgsConstructor
@Slf4j
public class OutboxPoller {
    

    private final OutboxRepository outboxRepository;
    private final KafkaTemplate<String, Object> kafkaTemplate;

    

    @Scheduled(fixedDelay = 1000)
    @Transactional
    public void publishPending() {
        List<OutboxEntity> rows = outboxRepository.findUnprocessed(PageRequest.of(0, 50));
        if (rows.isEmpty()) return;

        for (OutboxEntity row : rows) {
            try {
                kafkaTemplate.send("user.events", row.getAggregateId().toString(), row.getPayload()).get();
                row.setProcessedAt(OffsetDateTime.now());
            } catch (Exception ex) {
                log.warn("Outbox publish failed: {}", ex.getMessage());
            }
        }
    }
}

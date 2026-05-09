package com.goktug.conversation.outbox;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.domain.PageRequest;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.time.OffsetDateTime;
import java.util.List;

/**
 * Transactional Outbox Poller.
 *
 * Her N ms'de bir çalışır, işlenmemiş outbox satırlarını Kafka'ya basar.
 * Başarısız olursa attemptCount artar, lastError set edilir, satır kalır
 * (sonraki tick yeniden dener — exponential backoff TODO).
 *
 * Multi-replica safe: PESSIMISTIC_WRITE lock + transactional read sayesinde
 * iki instance aynı satırı işleyemez.
 *
 * Alternatif: Debezium CDC ile direkt Postgres WAL'i okumak — daha
 * verimli ama operasyonel karmaşıklık daha yüksek. Faz 3'te değerlendirilir.
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class OutboxPoller {

    private final OutboxRepository outboxRepository;
    private final KafkaTemplate<String, Object> kafkaTemplate;

    @Value("${outbox.batch-size:50}")
    private int batchSize;

    @Scheduled(fixedDelayString = "${outbox.poll-interval-ms:1000}")
    @Transactional
    public void publishPending() {
        List<OutboxEntity> rows = outboxRepository.findUnprocessed(PageRequest.of(0, batchSize));
        if (rows.isEmpty()) return;

        log.debug("Outbox: publishing {} events", rows.size());

        for (OutboxEntity row : rows) {
            try {
                kafkaTemplate.send(row.getTopic(), row.getAggregateId().toString(), row.getPayload())
                        .get();   // sync wait — basit; production'da async + callback
                row.setProcessedAt(OffsetDateTime.now());
            } catch (Exception ex) {
                log.warn("Outbox publish failed for id={} type={} attempt={}",
                        row.getId(), row.getEventType(), row.getAttemptCount() + 1, ex);
                row.setAttemptCount(row.getAttemptCount() + 1);
                row.setLastError(ex.getMessage());
                // satır işlenmemiş kalır → sonraki tick tekrar dener
            }
        }
        // @Transactional sayesinde JPA dirty checking otomatik update eder
    }
}

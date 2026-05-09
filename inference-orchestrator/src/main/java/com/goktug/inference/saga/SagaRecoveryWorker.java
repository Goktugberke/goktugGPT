package com.goktug.inference.saga;

import com.goktug.inference.client.BillingClient;
import com.goktug.inference.domain.InferenceJob;
import com.goktug.inference.domain.InferenceJobRepository;
import com.goktug.inference.event.InferenceEventPublisher;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.time.OffsetDateTime;
import java.util.List;

/**
 * Saga Recovery Worker.
 *
 * Problem: orchestrator pod'u STREAMING state'inde iken Ã§Ã¶kerse, DB'de
 * "yarÄ±m kalmÄ±ÅŸ" job'lar kalÄ±r â€” frontend SSE baÄŸlantÄ±sÄ± koptu, bir daha
 * cevap gelmeyecek.
 *
 * Ã‡Ã¶zÃ¼m: ayrÄ± bir scheduler her 30s'de stale job'larÄ± arar ve compensation
 * uygular:
 *   1. State INFERENCE_FAILED'a Ã§ek
 *   2. inference.failed.v1 event publish et (consumer'lar haberdar olsun)
 *   3. EÄŸer billing pre-deduct yapÄ±ldÄ±ysa refund tetikle
 *
 * "Stale" tanÄ±mÄ±: terminal olmayan + updated_at > 10 dk Ã¶nce.
 *
 * Multi-replica safe: @Transactional + JPA optimistic lock yeterli (her job'u
 * sadece bir replica iÅŸler â€” race ederse ikincisi state COMPLETED gÃ¶rÃ¼r).
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class SagaRecoveryWorker {

    private final InferenceJobRepository repository;
    private final InferenceEventPublisher publisher;
    private final BillingClient billing;

    @Value("${saga.recovery.stale-after-minutes:10}")
    private int staleAfterMinutes;

    @Value("${saga.recovery.batch-size:50}")
    private int batchSize;

    @Scheduled(fixedDelayString = "${saga.recovery.interval-ms:30000}")
    @Transactional
    public void recoverStaleJobs() {
        OffsetDateTime threshold = OffsetDateTime.now().minusMinutes(staleAfterMinutes);
        List<SagaState> nonTerminal = List.of(
            SagaState.PENDING,
            SagaState.GUARDRAIL_CHECK,
            SagaState.QUOTA_CHECK,
            SagaState.ROUTING,
            SagaState.STREAMING
        );

        List<InferenceJob> stale = repository.findStaleJobs(nonTerminal, threshold);
        if (stale.isEmpty()) return;

        log.warn("Saga recovery: found {} stale jobs (older than {}min)",
            stale.size(), staleAfterMinutes);

        int processed = 0;
        for (InferenceJob job : stale) {
            if (processed >= batchSize) break;
            try {
                String reason = "Saga recovery: orchestrator may have crashed in state " + job.getState();
                job.setState(SagaState.INFERENCE_FAILED);
                job.setErrorMessage(reason);
                job.setCompletedAt(OffsetDateTime.now());
                repository.save(job);

                publisher.publishFailed(job.getId(), job.getChatId(), job.getUserId(), reason);

                // Compensate: streaming'e baÅŸlamÄ±ÅŸ olabilir â†’ token refund
                billing.refund(job.getUserId(), job.getId(), 0).subscribe();

                log.info("Recovered stale job: id={} prevState={}", job.getId(), job.getState());
                processed++;
            } catch (Exception ex) {
                log.error("Failed to recover job {}", job.getId(), ex);
            }
        }
    }
}


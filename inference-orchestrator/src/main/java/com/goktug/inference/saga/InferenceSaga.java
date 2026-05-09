package com.goktug.inference.saga;

import com.goktug.inference.client.BillingClient;
import com.goktug.inference.client.GuardrailClient;
import com.goktug.inference.client.InferenceWorkerClient;
import com.goktug.inference.domain.InferenceJob;
import com.goktug.inference.domain.InferenceJobRepository;
import com.goktug.inference.event.InferenceEventPublisher;
import com.goktug.inference.client.RouterClient;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.OffsetDateTime;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Inference Saga (Orchestration tabanlı).
 *
 * Akış:
 *   1. PENDING            → DB'ye job kaydet
 *   2. GUARDRAIL_CHECK    → guardrail-service çağır
 *      └─ unsafe → BLOCKED (Mono.error → SSE error event → terminate)
 *   3. QUOTA_CHECK        → billing-service çağır
 *      └─ exceeded → QUOTA_EXCEEDED → SSE error event
 *   4. ROUTING            → Faz 2'de router-service. Faz 1'de hardcode "goktug-medium".
 *   5. STREAMING          → inference-worker çağır → token akışı
 *      └─ done → COMPLETED + Kafka publish
 *      └─ error → INFERENCE_FAILED + COMPENSATE (refund quota)
 *
 * Bu sınıf reactive (WebFlux Flux). Controller bu Flux'ı SSE olarak
 * frontend'e iletir.
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class InferenceSaga {

    private final InferenceJobRepository jobRepository;
    private final GuardrailClient guardrail;
    private final BillingClient billing;
    private final RouterClient router;
    private final InferenceWorkerClient worker;
    private final InferenceEventPublisher publisher;

    public Flux<TokenEvent> execute(InferenceRequest req) {
        UUID jobId = UUID.randomUUID();
        long startNanos = System.nanoTime();
        log.info("Saga[{}] start chat={} user={}", jobId, req.chatId(), req.userId());

        StringBuilder responseAccumulator = new StringBuilder();
        AtomicReference<String> selectedModel = new AtomicReference<>("goktug-medium");

        // Step 0: persist job
        return Mono.fromCallable(() -> persistJob(jobId, req, SagaState.PENDING))
            // Step 1: guardrail
            .flatMap(j -> {
                updateState(jobId, SagaState.GUARDRAIL_CHECK);
                return guardrail.check(req.text());
            })
            .flatMap(check -> {
                if (!check.safe()) {
                    return Mono.error(new SagaException("BLOCKED", check.reason()));
                }
                return Mono.just(check);
            })
            // Step 2: quota
            .flatMap(_g -> {
                updateState(jobId, SagaState.QUOTA_CHECK);
                return billing.checkQuota(req.userId());
            })
            .flatMap(quota -> {
                if (!quota.ok()) {
                    return Mono.error(new SagaException("QUOTA_EXCEEDED", "Quota exceeded"));
                }
                return Mono.just(quota);
            })
            // Step 3: routing
            .flatMap(_q -> {
                updateState(jobId, SagaState.ROUTING);
                return router.pickModel(req.text());
            })
            .flatMap(route -> {
                String model = route.model();
                selectedModel.set(model);
                return Mono.just(model);
            })
            // Step 4: streaming
            .flatMapMany(model -> {
                updateState(jobId, SagaState.STREAMING);
                InferenceWorkerClient.GenerateRequest workerReq =
                    new InferenceWorkerClient.GenerateRequest(
                        jobId, req.text(),
                        512, 0.7, 40, 0.9, 1.1);
                return worker.generateStream(workerReq);
            })
            .map(workerEvent -> {
                if ("token".equals(workerEvent.type())) {
                    responseAccumulator.append(workerEvent.content());
                }
                return new TokenEvent(jobId, workerEvent.type(),
                    workerEvent.content(),
                    workerEvent.tokenCount(),
                    null);
            })
            .doOnComplete(() -> {
                long latencyMs = (System.nanoTime() - startNanos) / 1_000_000;
                String fullText = responseAccumulator.toString();
                int approxTokens = Math.max(1, fullText.split("\\s+").length);
                completeSaga(jobId, fullText, selectedModel.get(),
                    approxTokens, approxTokens, latencyMs, req);
            })
            .onErrorResume(ex -> handleFailure(jobId, req, ex));
    }

    /**
     * Compensate + emit error event to client.
     */
    private Flux<TokenEvent> handleFailure(UUID jobId, InferenceRequest req, Throwable ex) {
        SagaState terminal = SagaState.INFERENCE_FAILED;
        String reason = ex.getMessage();

        if (ex instanceof SagaException se) {
            terminal = switch (se.code) {
                case "BLOCKED" -> SagaState.BLOCKED;
                case "QUOTA_EXCEEDED" -> SagaState.QUOTA_EXCEEDED;
                default -> SagaState.INFERENCE_FAILED;
            };
            reason = se.detail;
        }

        log.warn("Saga[{}] failed terminal={} reason={}", jobId, terminal, reason);

        markFailed(jobId, terminal, reason);
        publisher.publishFailed(jobId, req.chatId(), req.userId(), reason);

        // Compensate: streaming'e başladıktan sonra fail olduysa tokenları geri ver
        if (terminal == SagaState.INFERENCE_FAILED) {
            billing.refund(req.userId(), jobId, 0).subscribe();
        }

        return Flux.just(new TokenEvent(jobId, "error", reason, null, terminal.name()));
    }

    @Transactional
    public InferenceJob persistJob(UUID jobId, InferenceRequest req, SagaState state) {
        InferenceJob job = new InferenceJob();
        job.setId(jobId);
        job.setChatId(req.chatId());
        job.setUserId(req.userId());
        job.setUserMessageId(req.userMessageId());
        job.setState(state);
        return jobRepository.save(job);
    }

    @Async
    @Transactional
    public void updateState(UUID jobId, SagaState newState) {
        jobRepository.findById(jobId).ifPresent(job -> {
            job.setState(newState);
            jobRepository.save(job);
        });
    }

    @Async
    @Transactional
    public void markFailed(UUID jobId, SagaState terminal, String reason) {
        jobRepository.findById(jobId).ifPresent(job -> {
            job.setState(terminal);
            job.setErrorMessage(reason);
            job.setCompletedAt(OffsetDateTime.now());
            jobRepository.save(job);
        });
    }

    @Async
    @Transactional
    public void completeSaga(
            UUID jobId, String fullText, String modelUsed,
            int promptTokens, int completionTokens, long latencyMs,
            InferenceRequest req) {
        jobRepository.findById(jobId).ifPresent(job -> {
            job.setState(SagaState.COMPLETED);
            job.setModelUsed(modelUsed);
            job.setPromptTokens(promptTokens);
            job.setCompletionTokens(completionTokens);
            job.setCompletedAt(OffsetDateTime.now());
            jobRepository.save(job);
        });
        publisher.publishCompleted(jobId, req.chatId(), req.userId(),
            req.userMessageId(), fullText, modelUsed,
            promptTokens, completionTokens, latencyMs);
    }

    public record InferenceRequest(
        UUID chatId,
        UUID userId,
        UUID userMessageId,
        String text,
        String modelHint
    ) {}

    public record TokenEvent(UUID jobId, String type, String content, Integer tokenCount, String terminalState) {}

    private static class SagaException extends RuntimeException {
        final String code;
        final String detail;
        SagaException(String code, String detail) {
            super(code + ": " + detail);
            this.code = code;
            this.detail = detail;
        }
    }
}

package com.goktug.billing.service;

import com.goktug.billing.config.FreePlanProperties;
import com.goktug.billing.domain.*;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.OffsetDateTime;
import java.util.Optional;
import java.util.UUID;

@Service
@RequiredArgsConstructor
@Slf4j
public class BillingService {

    private final PlanRepository planRepository;
    private final SubscriptionRepository subscriptionRepository;
    private final UsageRecordRepository usageRecordRepository;
    private final FreePlanProperties freePlanProps;

    /**
     * Ensure user has an active subscription. Creates a free plan subscription
     * if none exists. Idempotent — duplicate user.registered.v1 events are safe.
     */
    @Transactional
    public SubscriptionEntity ensureFreeSubscription(UUID userId) {
        Optional<SubscriptionEntity> existing = subscriptionRepository.findByUserId(userId);
        if (existing.isPresent()) {
            return existing.get();
        }

        PlanEntity freePlan = planRepository.findByName(freePlanProps.getName())
            .orElseThrow(() -> new IllegalStateException(
                "Free plan not seeded: " + freePlanProps.getName()));

        OffsetDateTime now = OffsetDateTime.now();
        SubscriptionEntity sub = SubscriptionEntity.builder()
            .userId(userId)
            .planId(freePlan.getId())
            .status("ACTIVE")
            .periodStart(now)
            .periodEnd(now.plusMonths(1))
            .tokensUsed(0)
            .build();

        SubscriptionEntity saved = subscriptionRepository.save(sub);
        log.info("Created free subscription for user={} subId={}", userId, saved.getId());
        return saved;
    }

    /**
     * Record token usage. Idempotent on jobId — duplicate consumption of
     * inference.completed.v1 will not double-charge.
     */
    @Transactional
    public void recordUsage(UUID userId, UUID jobId, String model,
                            int promptTokens, int completionTokens, Integer latencyMs) {
        if (jobId != null && usageRecordRepository.existsByJobId(jobId)) {
            log.debug("Usage already recorded for jobId={}, skipping", jobId);
            return;
        }

        SubscriptionEntity sub = subscriptionRepository.findByUserId(userId)
            .orElseGet(() -> ensureFreeSubscription(userId));

        int total = promptTokens + completionTokens;
        UsageRecordEntity record = UsageRecordEntity.builder()
            .userId(userId)
            .subscriptionId(sub.getId())
            .jobId(jobId)
            .model(model)
            .promptTokens(promptTokens)
            .completionTokens(completionTokens)
            .totalTokens(total)
            .latencyMs(latencyMs)
            .build();
        usageRecordRepository.save(record);

        subscriptionRepository.incrementTokensUsed(userId, total);
        log.info("Recorded usage user={} job={} tokens={}", userId, jobId, total);
    }

    @Transactional(readOnly = true)
    public QuotaSummary getQuota(UUID userId) {
        SubscriptionEntity sub = subscriptionRepository.findByUserId(userId)
            .orElseGet(() -> ensureFreeSubscription(userId));
        PlanEntity plan = planRepository.findById(sub.getPlanId())
            .orElseThrow(() -> new IllegalStateException("Plan missing: " + sub.getPlanId()));
        long remaining = Math.max(0, plan.getMonthlyTokenQuota() - sub.getTokensUsed());
        return new QuotaSummary(
            plan.getName(),
            plan.getMonthlyTokenQuota(),
            sub.getTokensUsed(),
            remaining,
            sub.getPeriodEnd()
        );
    }

    @Transactional(readOnly = true)
    public Optional<SubscriptionEntity> findSubscription(UUID userId) {
        return subscriptionRepository.findByUserId(userId);
    }

    public record QuotaSummary(
        String plan,
        long monthlyQuota,
        long tokensUsed,
        long tokensRemaining,
        OffsetDateTime periodEnd
    ) {}
}

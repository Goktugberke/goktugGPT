package com.goktug.billing.api;

import com.goktug.billing.config.FreePlanProperties;
import com.goktug.billing.domain.SubscriptionEntity;
import com.goktug.billing.service.BillingService;
import com.goktug.billing.service.TokenBucketService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;
import java.util.UUID;

@RestController
@RequestMapping("/api/v1/billing")
@RequiredArgsConstructor
public class BillingController {

    private final BillingService billingService;
    private final TokenBucketService tokenBucketService;
    private final FreePlanProperties freePlanProps;

    /**
     * Saga step: pre-flight rate-limit + remaining-quota check.
     * Body: { userId, tokens? }
     * Response: 200 { allowed: true, remaining: N } / 429 { allowed: false, reason }
     */
    @PostMapping("/quota/check")
    public ResponseEntity<?> checkQuota(@RequestBody Map<String, Object> request) {
        UUID userId = UUID.fromString((String) request.get("userId"));
        int requestedTokens = ((Number) request.getOrDefault("tokens", 1)).intValue();

        BillingService.QuotaSummary quota = billingService.getQuota(userId);
        if (quota.tokensRemaining() < requestedTokens) {
            return ResponseEntity.status(429).body(Map.of(
                "allowed", false,
                "reason", "Monthly token quota exceeded",
                "remaining", quota.tokensRemaining()
            ));
        }

        boolean rateOk = tokenBucketService.tryConsume(
            userId.toString(), requestedTokens,
            freePlanProps.getRateLimitCapacity(),
            freePlanProps.getRateLimitRefillPerSec()
        );
        if (!rateOk) {
            return ResponseEntity.status(429).body(Map.of(
                "allowed", false,
                "reason", "Rate limit exceeded"
            ));
        }

        return ResponseEntity.ok(Map.of(
            "allowed", true,
            "remaining", quota.tokensRemaining()
        ));
    }

    /** Current user's quota summary. */
    @GetMapping("/me/quota")
    public ResponseEntity<BillingService.QuotaSummary> myQuota(
        @RequestHeader("X-User-Id") UUID userId
    ) {
        return ResponseEntity.ok(billingService.getQuota(userId));
    }

    /** Current user's subscription (creates free plan if absent). */
    @GetMapping("/me/subscription")
    public ResponseEntity<SubscriptionEntity> mySubscription(
        @RequestHeader("X-User-Id") UUID userId
    ) {
        SubscriptionEntity sub = billingService.findSubscription(userId)
            .orElseGet(() -> billingService.ensureFreeSubscription(userId));
        return ResponseEntity.ok(sub);
    }
}

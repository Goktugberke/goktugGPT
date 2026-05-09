package com.goktug.billing.api;

import com.goktug.billing.service.TokenBucketService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/v1/billing")
@RequiredArgsConstructor
public class BillingController {

    private final TokenBucketService tokenBucketService;

    

    @PostMapping("/quota/check")
    public ResponseEntity<?> checkQuota(@RequestBody Map<String, Object> request) {
        String userId = (String) request.get("userId");
        int tokens = (int) request.getOrDefault("tokens", 1);
        
        // Default limits: 1000 tokens capacity, 10 tokens/sec refill
        boolean allowed = tokenBucketService.tryConsume(userId, tokens, 1000, 10.0);
        
        if (allowed) {
            return ResponseEntity.ok(Map.of("allowed", true));
        } else {
            return ResponseEntity.status(429).body(Map.of("allowed", false, "reason", "Quota exceeded"));
        }
    }
}

package com.goktug.billing.service;

import lombok.RequiredArgsConstructor;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.script.DefaultRedisScript;
import org.springframework.stereotype.Service;

import java.util.Collections;
import java.util.List;

@Service
@RequiredArgsConstructor
public class TokenBucketService {

    private final StringRedisTemplate redisTemplate;

    // Lua script for atomic token bucket check-and-decrement
    private static final String LUA_SCRIPT =
        "local bucket_key = KEYS[1]\n" +
        "local tokens_to_consume = tonumber(ARGV[1])\n" +
        "local capacity = tonumber(ARGV[2])\n" +
        "local refill_rate = tonumber(ARGV[3])\n" +
        "local now = tonumber(ARGV[4])\n" +
        "\n" +
        "local bucket = redis.call('hmget', bucket_key, 'tokens', 'last_refill')\n" +
        "local tokens = tonumber(bucket[1])\n" +
        "local last_refill = tonumber(bucket[2])\n" +
        "\n" +
        "if not tokens then\n" +
        "  tokens = capacity\n" +
        "  last_refill = now\n" +
        "else\n" +
        "  local elapsed = math.max(0, now - last_refill)\n" +
        "  tokens = math.min(capacity, tokens + (elapsed * refill_rate))\n" +
        "  last_refill = now\n" +
        "end\n" +
        "\n" +
        "if tokens >= tokens_to_consume then\n" +
        "  tokens = tokens - tokens_to_consume\n" +
        "  redis.call('hmset', bucket_key, 'tokens', tokens, 'last_refill', last_refill)\n" +
        "  return 1\n" +
        "else\n" +
        "  redis.call('hmset', bucket_key, 'tokens', tokens, 'last_refill', last_refill)\n" +
        "  return 0\n" +
        "end";

    

    public boolean tryConsume(String userId, int tokens, int capacity, double refillRate) {
        String key = "ratelimit:" + userId;
        Long result = redisTemplate.execute(
            new DefaultRedisScript<>(LUA_SCRIPT, Long.class),
            Collections.singletonList(key),
            String.valueOf(tokens),
            String.valueOf(capacity),
            String.valueOf(refillRate),
            String.valueOf(System.currentTimeMillis() / 1000)
        );
        return result != null && result == 1;
    }
}

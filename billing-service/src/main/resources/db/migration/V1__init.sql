-- billing-service initial schema

CREATE TABLE plans (
    id              UUID PRIMARY KEY,
    name            VARCHAR(64) NOT NULL UNIQUE,
    monthly_token_quota BIGINT NOT NULL,
    rate_limit_capacity INTEGER NOT NULL,
    rate_limit_refill_per_sec DOUBLE PRECISION NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE subscriptions (
    id              UUID PRIMARY KEY,
    user_id         UUID NOT NULL UNIQUE,
    plan_id         UUID NOT NULL REFERENCES plans(id),
    status          VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
    period_start    TIMESTAMPTZ NOT NULL,
    period_end      TIMESTAMPTZ NOT NULL,
    tokens_used     BIGINT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_subscriptions_user ON subscriptions(user_id);

CREATE TABLE usage_records (
    id              UUID PRIMARY KEY,
    user_id         UUID NOT NULL,
    subscription_id UUID NOT NULL REFERENCES subscriptions(id),
    job_id          UUID,
    model           VARCHAR(64),
    prompt_tokens   INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens    INTEGER NOT NULL,
    latency_ms      INTEGER,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_usage_user_date ON usage_records(user_id, recorded_at DESC);
CREATE UNIQUE INDEX idx_usage_job_dedup ON usage_records(job_id) WHERE job_id IS NOT NULL;

-- Seed default free plan
INSERT INTO plans (id, name, monthly_token_quota, rate_limit_capacity, rate_limit_refill_per_sec)
VALUES (
    '00000000-0000-0000-0000-000000000001',
    'free',
    100000,
    1000,
    10.0
);

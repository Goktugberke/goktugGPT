-- inference-orchestrator schema v1

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Saga state persistence — orchestrator restart olursa in-flight saga'lar
-- recovery worker tarafından yeniden başlatılabilir.
CREATE TABLE inference_jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chat_id         UUID NOT NULL,
    user_id         UUID NOT NULL,
    user_message_id UUID NOT NULL,
    state           VARCHAR(32) NOT NULL,         -- SagaState enum
    model_used      VARCHAR(64),
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    error_message   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);

CREATE INDEX idx_jobs_state ON inference_jobs (state) WHERE state NOT IN ('COMPLETED', 'BLOCKED', 'QUOTA_EXCEEDED', 'INFERENCE_FAILED');
CREATE INDEX idx_jobs_user ON inference_jobs (user_id, created_at DESC);

-- Saga step audit log — debugging + compliance için
CREATE TABLE saga_steps (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id      UUID NOT NULL REFERENCES inference_jobs(id) ON DELETE CASCADE,
    step_name   VARCHAR(64) NOT NULL,
    status      VARCHAR(16) NOT NULL,             -- STARTED, OK, FAIL, COMPENSATED
    request     JSONB,
    response    JSONB,
    duration_ms INTEGER,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_saga_steps_job ON saga_steps (job_id, created_at);

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE assets (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL,
    chat_id         UUID,
    original_name   VARCHAR(512) NOT NULL,
    mime_type       VARCHAR(120) NOT NULL,
    size_bytes      BIGINT NOT NULL,
    storage_path    VARCHAR(1024) NOT NULL,    -- bucket içindeki obje yolu
    checksum_sha256 VARCHAR(64),
    status          VARCHAR(16) NOT NULL,      -- PENDING_UPLOAD, READY, DELETED
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    confirmed_at    TIMESTAMPTZ,
    deleted_at      TIMESTAMPTZ
);

CREATE INDEX idx_assets_user ON assets (user_id, created_at DESC);
CREATE INDEX idx_assets_chat ON assets (chat_id) WHERE chat_id IS NOT NULL;

-- Outbox (asset.uploaded.v1, asset.deleted.v1)
CREATE TABLE outbox (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    aggregate_id  UUID NOT NULL,
    event_type    VARCHAR(120) NOT NULL,
    topic         VARCHAR(120) NOT NULL,
    payload       JSONB NOT NULL,
    headers       JSONB,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at  TIMESTAMPTZ,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_error    TEXT
);

CREATE INDEX idx_outbox_unprocessed ON outbox (created_at) WHERE processed_at IS NULL;

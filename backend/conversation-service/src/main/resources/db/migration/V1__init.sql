-- conversation-service schema v1
--
-- Database-per-service kuralı: bu DB'ye sadece conversation-service erişebilir.
-- Diğer servisler veriyi REST API'den ister.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================================
-- chats
-- ============================================================
CREATE TABLE chats (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID NOT NULL,
    title        VARCHAR(256),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at   TIMESTAMPTZ
);

CREATE INDEX idx_chats_user_updated ON chats (user_id, updated_at DESC) WHERE deleted_at IS NULL;

-- ============================================================
-- messages — sender enum: USER | ASSISTANT | SYSTEM
-- ============================================================
CREATE TABLE messages (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chat_id       UUID NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    sender        VARCHAR(16) NOT NULL,
    content       TEXT NOT NULL,
    model_used    VARCHAR(64),
    token_count   INTEGER,
    attachments   JSONB,                            -- asset-service'teki ID'ler
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_messages_chat ON messages (chat_id, created_at);

-- ============================================================
-- Idempotency Pattern
-- POST endpointleri için: aynı (user_id, idempotency_key) tekrar gelirse
-- önceki response'u döndür, yeniden çalıştırma.
-- ============================================================
CREATE TABLE idempotency_keys (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL,
    idempotency_key VARCHAR(120) NOT NULL,
    endpoint        VARCHAR(120) NOT NULL,
    response_body   JSONB,
    response_status INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '24 hours'),
    UNIQUE (user_id, idempotency_key, endpoint)
);

CREATE INDEX idx_idempotency_expires ON idempotency_keys (expires_at);

-- ============================================================
-- Transactional Outbox Pattern
-- Mesaj DB'ye yazılırken aynı transaction'da outbox row insert edilir.
-- Ayrı bir poller (OutboxPoller) bunu Kafka'ya basar, processed_at işaretler.
-- Böylece DB-Kafka tutarlılığı garantili (at-least-once).
-- ============================================================
CREATE TABLE outbox (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    aggregate_id  UUID NOT NULL,                    -- chatId / messageId
    event_type    VARCHAR(120) NOT NULL,            -- 'message.user-sent.v1'
    topic         VARCHAR(120) NOT NULL,            -- 'message.events'
    payload       JSONB NOT NULL,
    headers       JSONB,                            -- traceId, userId vs.
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at  TIMESTAMPTZ,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_error    TEXT
);

CREATE INDEX idx_outbox_unprocessed ON outbox (created_at) WHERE processed_at IS NULL;

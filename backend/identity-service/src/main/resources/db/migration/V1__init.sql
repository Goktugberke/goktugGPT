-- identity-service schema v1
--
-- Bu DB Keycloak'tan FARKLI: Keycloak yalnız credential + JWT issuance yapar.
-- identity-service burada uygulama-spesifik profil verilerini tutar.
-- userId Keycloak'taki "sub" claim'iyle aynıdır (UUID).

CREATE TABLE profiles (
    user_id              UUID PRIMARY KEY,
    email                VARCHAR(320) NOT NULL UNIQUE,
    display_name         VARCHAR(120),
    avatar_url           VARCHAR(1024),
    language             VARCHAR(8) NOT NULL DEFAULT 'tr',
    theme                VARCHAR(16) NOT NULL DEFAULT 'system',
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_profiles_email ON profiles (email);

CREATE TABLE custom_instructions (
    user_id        UUID PRIMARY KEY REFERENCES profiles(user_id) ON DELETE CASCADE,
    -- "Modele her isteğe gizlice eklenen kullanıcı talimatları"
    about_user     TEXT,    -- "I'm a software engineer..."
    response_style TEXT,    -- "Be concise. Use bullet points."
    enabled        BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Outbox pattern: profile-updated, user-deleted gibi eventler için
CREATE TABLE outbox (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    aggregate_id  UUID NOT NULL,
    event_type    VARCHAR(120) NOT NULL,
    payload       JSONB NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at  TIMESTAMPTZ
);

CREATE INDEX idx_outbox_unprocessed ON outbox (created_at) WHERE processed_at IS NULL;

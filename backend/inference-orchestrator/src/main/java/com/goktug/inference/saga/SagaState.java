package com.goktug.inference.saga;

/**
 * Inference Saga state machine.
 *
 * Bir kullanıcı mesajı geldiğinde aşağıdaki adımlardan geçer:
 *
 *   PENDING                → saga oluşturuldu, henüz adım yok
 *      │
 *   GUARDRAIL_CHECK        → guardrail-service'e prompt gönderildi
 *      ├── ok →
 *      └── unsafe → BLOCKED (compensate yok — sadece error response)
 *      │
 *   QUOTA_CHECK            → billing-service'e quota sorgusu
 *      ├── ok →
 *      └── exceeded → QUOTA_EXCEEDED
 *      │
 *   ROUTING                → router-service hangi modeli seçti
 *      │
 *   STREAMING              → inference-worker'a stream call başladı
 *      ├── done →
 *      └── error → INFERENCE_FAILED
 *      │              ├── compensate: quotayı geri ver (billing.refund)
 *      │              └── compensate: pending message'ı failed işaretle
 *      │
 *   COMPLETED              → final state, inference.completed.v1 publish
 *
 * State `inference_jobs` tablosunda persist edilir → orchestrator restart
 * olsa bile in-flight saga'lar resume edilebilir (recovery worker).
 */
public enum SagaState {
    PENDING,
    GUARDRAIL_CHECK,
    QUOTA_CHECK,
    ROUTING,
    STREAMING,
    COMPLETED,

    // failure terminal states
    BLOCKED,           // guardrail unsafe
    QUOTA_EXCEEDED,
    INFERENCE_FAILED;

    public boolean isTerminal() {
        return this == COMPLETED
            || this == BLOCKED
            || this == QUOTA_EXCEEDED
            || this == INFERENCE_FAILED;
    }
}

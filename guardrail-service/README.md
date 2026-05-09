# guardrail-service  [Faz 2]

> **Prompt güvenlik filtresi.** Kullanıcı prompt'u ana modele gitmeden önce burada hızlı NLP modelleriyle taranır.

## Sorumluluklar

- **Prompt injection detection:** "ignore previous instructions" gibi manipülasyon girişimleri (DeBERTa-v3 fine-tune)
- **Toxicity:** Detoxify modeli ile zararlı içerik
- **PII detection:** TCKN, kredi kartı, e-posta gibi kişisel veri (Presidio veya regex+ML hibrit)
- **Jailbreak patterns:** known prompt jailbreak corpus (regex + similarity)

Synchronous endpoint — inference-orchestrator hot path'te çağırır, latency hedefi: **<200ms**.

## Mimari Karar

- **Python + FastAPI** çünkü tüm modeller HuggingFace transformers ekosisteminde.
- **In-memory model:** request başına load yok; startup'ta yüklenip RAM'de tutulur.
- **Stateless:** her request bağımsız → horizontal autoscale kolay.
- **No external API:** OpenAI Moderation API gibi paid çözümler değil — privacy + maliyet.

## Endpoint

```
POST /v1/check
Body: { "text": "..." }

Response:
{
  "safe": true|false,
  "categories": {
    "prompt_injection": 0.02,
    "toxicity":         0.01,
    "pii":              0.00
  },
  "blocked_reason": null,           // veya "prompt_injection_detected"
  "redacted_text":  "..."           // PII bulundıysa maskelenmiş hali
}
```

## Port

`9003` (internal — sadece inference-orchestrator çağırır)

## TODO

1. `app/main.py` FastAPI iskelet (inference-worker-goktug'a benzer pattern)
2. Model loading: DeBERTa (prompt injection) + Detoxify (toxicity)
3. PII detector: regex (TCKN, kredi kartı) + ner spaCy (isim, adres)
4. Prometheus metrics (block rate per category)
5. Caching: aynı text için son 1000 sonucu LRU cache (orchestrator retry'ları)
6. Threshold konfigi `config.yaml`: `safe_threshold: 0.5` per category
7. Test corpus: known jailbreak prompts + benign prompts → precision/recall ölç

## Faz Önceliği

**Faz 2 başlangıcı.** Faz 1 MVP'de orchestrator bu adımı bypass edebilir (`spring.profiles.active=dev` ile).

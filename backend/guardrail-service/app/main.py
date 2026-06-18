"""
guardrail-service — prompt safety classifier.

Checks each prompt for:
  - PII (Turkish TCKN, US SSN, credit card patterns, email, phone)
  - Toxicity (Detoxify, optional)
  - Prompt injection (DeBERTa-based classifier, optional)

Fail-safe: any internal error returns `safe=true` with category `degraded` —
inference SHOULD continue when guardrail is unavailable, because blocking
on degraded guardrails creates a single point of failure for the whole platform.
Set `FAIL_CLOSED=true` to invert this (block instead) for high-risk deployments.
"""

import logging
import os
import re
import time
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("guardrail")

USE_ML = os.getenv("USE_ML_MODELS", "false").lower() == "true"
FAIL_CLOSED = os.getenv("FAIL_CLOSED", "false").lower() == "true"
TOXICITY_THRESHOLD = float(os.getenv("TOXICITY_THRESHOLD", "0.7"))
INJECTION_THRESHOLD = float(os.getenv("INJECTION_THRESHOLD", "0.8"))

CHECK_LATENCY = Histogram("guardrail_check_seconds", "Guardrail check latency")
CHECK_COUNT = Counter("guardrail_checks_total", "Total checks", ["result"])

app = FastAPI(title="guardrail-service", version="0.2.0")


# ---------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------

PII_PATTERNS = {
    "ssn":         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "tckn":        re.compile(r"\b[1-9]\d{10}\b"),   # Turkish national ID (11 digits, no leading 0)
    "email":       re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone":       re.compile(r"\b\+?\d{1,3}[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2,4}\b"),
    "iban":        re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b"),
}


# ---------------------------------------------------------------------
# Keyword-based prompt injection fallback (no ML)
# ---------------------------------------------------------------------

INJECTION_PHRASES = [
    "ignore previous instructions",
    "ignore the above",
    "disregard your instructions",
    "you are now",
    "act as if",
    "reveal your system prompt",
    "show me your prompt",
    "print your instructions",
    "jailbreak",
    "developer mode",
    "dan mode",
    "önceki talimatları yoksay",
    "talimatları unut",
    "sistem promptunu göster",
]


# ---------------------------------------------------------------------
# Optional ML models
# ---------------------------------------------------------------------

class ModelBundle:
    def __init__(self):
        self.detoxify = None
        self.injection_pipeline = None

    def load(self):
        if not USE_ML:
            log.info("USE_ML_MODELS=false — running with rule-based checks only")
            return
        try:
            from detoxify import Detoxify
            self.detoxify = Detoxify("original")
            log.info("Detoxify loaded")
        except Exception as e:
            log.warning("Detoxify unavailable: %s", e)
        try:
            from transformers import pipeline
            self.injection_pipeline = pipeline(
                "text-classification",
                model="protectai/deberta-v3-base-prompt-injection",
            )
            log.info("Prompt injection classifier loaded")
        except Exception as e:
            log.warning("Prompt injection classifier unavailable: %s", e)


MODELS = ModelBundle()


@app.on_event("startup")
def _startup():
    MODELS.load()


# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------

class CheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20_000)


class CheckResponse(BaseModel):
    safe: bool
    categories: List[str] = []
    score: float = 0.0
    degraded: bool = False
    detail: Optional[str] = None


# ---------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------

def detect_pii(text: str) -> List[str]:
    return [name for name, pat in PII_PATTERNS.items() if pat.search(text)]


def detect_injection_heuristic(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in INJECTION_PHRASES)


def detect_toxicity_ml(text: str) -> float:
    if MODELS.detoxify is None:
        return 0.0
    try:
        results = MODELS.detoxify.predict(text)
        return float(results.get("toxicity", 0.0))
    except Exception as e:
        log.warning("Detoxify predict failed: %s", e)
        return 0.0


def detect_injection_ml(text: str) -> float:
    if MODELS.injection_pipeline is None:
        return 0.0
    try:
        results = MODELS.injection_pipeline(text[:512])
        if results and results[0]["label"].upper() == "INJECTION":
            return float(results[0]["score"])
        return 0.0
    except Exception as e:
        log.warning("Injection pipeline failed: %s", e)
        return 0.0


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------

@app.post("/v1/check", response_model=CheckResponse)
def check(req: CheckRequest):
    start = time.time()
    text = req.text
    categories: List[str] = []
    score = 0.0
    degraded = False

    try:
        pii = detect_pii(text)
        if pii:
            categories.extend(f"pii.{p}" for p in pii)
            score = max(score, 1.0)

        if detect_injection_heuristic(text):
            categories.append("prompt_injection.heuristic")
            score = max(score, 0.95)

        if USE_ML:
            tox = detect_toxicity_ml(text)
            if tox >= TOXICITY_THRESHOLD:
                categories.append("toxicity")
                score = max(score, tox)
            inj = detect_injection_ml(text)
            if inj >= INJECTION_THRESHOLD:
                categories.append("prompt_injection.ml")
                score = max(score, inj)

        safe = len(categories) == 0
        CHECK_COUNT.labels("safe" if safe else "blocked").inc()
        return CheckResponse(safe=safe, categories=categories, score=score)

    except Exception as e:
        log.exception("Guardrail internal error: %s", e)
        CHECK_COUNT.labels("degraded").inc()
        # Fail-safe (or fail-closed if configured)
        if FAIL_CLOSED:
            return CheckResponse(
                safe=False,
                categories=["degraded"],
                score=1.0,
                degraded=True,
                detail="Guardrail unavailable, blocking by policy (FAIL_CLOSED=true)",
            )
        return CheckResponse(
            safe=True,
            categories=["degraded"],
            score=0.0,
            degraded=True,
            detail="Guardrail unavailable, allowing by policy",
        )
    finally:
        CHECK_LATENCY.observe(time.time() - start)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "use_ml": USE_ML,
        "detoxify_loaded": MODELS.detoxify is not None,
        "injection_loaded": MODELS.injection_pipeline is not None,
        "fail_closed": FAIL_CLOSED,
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

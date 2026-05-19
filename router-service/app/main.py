"""
router-service — semantic prompt classifier.

Given a user prompt, decides which model variant should answer:
  - goktug-mini   : general chat, small/cheap
  - goktug-pro    : code, programming
  - goktug-reason : math, logic, reasoning
  - external-claude : fallback for hard tasks (Faz 3 — disabled here)

Two modes:
  - USE_EMBEDDINGS=false (default) : keyword heuristics, ~1 ms latency, no model
  - USE_EMBEDDINGS=true            : sentence-transformers nearest-centroid over
                                     pre-computed class anchors. Single in-memory
                                     classification cache keyed by prompt hash.

The hot-path budget is <50 ms (see MICROSERVICES_PLAN.md §2.7).
"""

import hashlib
import logging
import os
import time
from typing import List, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("router")

USE_EMBEDDINGS = os.getenv("USE_EMBEDDINGS", "false").lower() == "true"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "1024"))

ROUTE_LATENCY = Histogram("router_route_seconds", "Routing latency",
                          buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25))
ROUTE_DECISION = Counter("router_decisions_total", "Routing decisions", ["model"])

app = FastAPI(title="router-service", version="0.2.0")


# ---------------------------------------------------------------------
# Class anchors — each class has a list of representative prompts.
# Embedding mode averages their embeddings; heuristic mode uses keywords.
# ---------------------------------------------------------------------

CLASS_ANCHORS = {
    "goktug-pro": {
        "anchors": [
            "Write a Python function to sort a list",
            "Explain this Java code",
            "Debug this JavaScript",
            "How do I implement a linked list in C++",
            "Refactor this function to be more efficient",
            "Fix this TypeScript error",
        ],
        "keywords": [
            "code", "function", "python", "java", "javascript", "typescript",
            "c++", "rust", "go ", "kotlin", "debug", "refactor", "compile",
            "syntax", "regex", "api", "sql query", "implement",
        ],
    },
    "goktug-reason": {
        "anchors": [
            "If a train travels 60 mph for 3 hours, how far does it go?",
            "Solve this equation step by step: 2x + 5 = 17",
            "Why is the sky blue?",
            "Prove that the sum of two odd numbers is even",
            "What is the derivative of x squared?",
            "How many primes are less than 100?",
        ],
        "keywords": [
            "solve", "calculate", "prove", "derive", "math", "equation",
            "logic", "puzzle", "step by step", "reasoning", "geometry",
            "algebra", "calculus", "integral", "derivative", "theorem",
        ],
    },
    "goktug-mini": {
        "anchors": [
            "Hello, how are you?",
            "Tell me a joke",
            "What's your name?",
            "Recommend a good movie",
            "Translate this to Turkish",
            "Summarize this paragraph",
        ],
        "keywords": [],   # fallback class
    },
}

DEFAULT_MODEL = "goktug-mini"


# ---------------------------------------------------------------------
# Heuristic classifier
# ---------------------------------------------------------------------

def classify_heuristic(text: str) -> Tuple[str, float]:
    lower = text.lower()
    best_model = DEFAULT_MODEL
    best_hits = 0
    for model, conf in CLASS_ANCHORS.items():
        hits = sum(1 for kw in conf["keywords"] if kw in lower)
        if hits > best_hits:
            best_hits = hits
            best_model = model
    # Confidence proportional to keyword hits, clipped to [0.5, 0.95]
    confidence = min(0.95, 0.5 + 0.15 * best_hits) if best_hits > 0 else 0.6
    return best_model, confidence


# ---------------------------------------------------------------------
# Embedding classifier
# ---------------------------------------------------------------------

class EmbeddingClassifier:
    def __init__(self):
        self.model = None
        self.centroids: dict = {}
        self.cache: dict = {}

    def load(self):
        if not USE_EMBEDDINGS:
            return
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            self._np = np
            for model_name, conf in CLASS_ANCHORS.items():
                anchors = conf["anchors"]
                if not anchors:
                    continue
                embeddings = self.model.encode(anchors, normalize_embeddings=True)
                centroid = embeddings.mean(axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                self.centroids[model_name] = centroid
            log.info("Embedding classifier ready (model=%s, classes=%d)",
                     EMBEDDING_MODEL, len(self.centroids))
        except Exception as e:
            log.warning("Embedding model load failed (%s) — falling back to heuristics", e)
            self.model = None

    def classify(self, text: str) -> Optional[Tuple[str, float]]:
        if self.model is None:
            return None

        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if key in self.cache:
            return self.cache[key]

        try:
            np = self._np
            embedding = self.model.encode(text, normalize_embeddings=True)
            best_model, best_sim = DEFAULT_MODEL, -1.0
            for model_name, centroid in self.centroids.items():
                sim = float(np.dot(embedding, centroid))
                if sim > best_sim:
                    best_sim = sim
                    best_model = model_name
            confidence = (best_sim + 1) / 2   # cosine [-1,1] → [0,1]
            result = (best_model, confidence)

            if len(self.cache) >= CACHE_SIZE:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = result
            return result
        except Exception as e:
            log.warning("Embedding classify failed: %s", e)
            return None


CLASSIFIER = EmbeddingClassifier()


@app.on_event("startup")
def _startup():
    CLASSIFIER.load()


# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------

class RouteRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=20_000)
    user_hint: Optional[str] = None


class RouteResponse(BaseModel):
    model: str
    confidence: float
    method: str   # "heuristic" | "embedding" | "hint"
    candidates: List[str] = []


# ---------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------

@app.post("/v1/route", response_model=RouteResponse)
def route(req: RouteRequest):
    start = time.time()
    try:
        if req.user_hint and req.user_hint in CLASS_ANCHORS:
            ROUTE_DECISION.labels(req.user_hint).inc()
            return RouteResponse(
                model=req.user_hint, confidence=1.0, method="hint",
                candidates=list(CLASS_ANCHORS.keys()),
            )

        result = CLASSIFIER.classify(req.prompt)
        if result is not None:
            model, conf = result
            ROUTE_DECISION.labels(model).inc()
            return RouteResponse(
                model=model, confidence=conf, method="embedding",
                candidates=list(CLASS_ANCHORS.keys()),
            )

        model, conf = classify_heuristic(req.prompt)
        ROUTE_DECISION.labels(model).inc()
        return RouteResponse(
            model=model, confidence=conf, method="heuristic",
            candidates=list(CLASS_ANCHORS.keys()),
        )
    finally:
        ROUTE_LATENCY.observe(time.time() - start)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "use_embeddings": USE_EMBEDDINGS,
        "model_loaded": CLASSIFIER.model is not None,
        "cache_size": len(CLASSIFIER.cache),
        "classes": list(CLASS_ANCHORS.keys()),
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

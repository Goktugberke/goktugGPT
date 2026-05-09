from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import re
import logging

app = FastAPI(title="Guardrail Service")
logger = logging.getLogger("guardrail")

class CheckRequest(BaseModel):
    text: str

class CheckResponse(BaseModel):
    safe: bool
    categories: List[str]
    score: float

# A simple rule-based PII regex check as a placeholder,
# and Detoxify for toxicity check if available.
try:
    from detoxify import Detoxify
    detoxifier = Detoxify('original')
except ImportError:
    detoxifier = None
    logger.warning("Detoxify not available. Falling back to basic regex guardrail.")

PII_PATTERN = re.compile(r'\b(?:\d{3}-\d{2}-\d{4}|\d{16})\b')

@app.post("/v1/check", response_model=CheckResponse)
async def check_prompt(req: CheckRequest):
    text = req.text
    categories = []
    score = 0.0
    safe = True
    
    # Check for PII (dummy rule)
    if PII_PATTERN.search(text):
        safe = False
        categories.append("pii")
        score = max(score, 1.0)
        
    # Check toxicity using Detoxify
    if detoxifier is not None:
        try:
            results = detoxifier.predict(text)
            toxicity_score = results.get("toxicity", 0.0)
            if toxicity_score > 0.7:
                safe = False
                categories.append("toxic")
                score = max(score, float(toxicity_score))
        except Exception as e:
            logger.error(f"Error checking toxicity: {e}")
            
    return CheckResponse(safe=safe, categories=categories, score=score)

@app.get("/health")
async def health():
    return {"status": "ok"}

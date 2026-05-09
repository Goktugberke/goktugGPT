from fastapi import FastAPI
from pydantic import BaseModel
import logging

app = FastAPI(title="Router Service")
logger = logging.getLogger("router")

class RouteRequest(BaseModel):
    prompt: str

class RouteResponse(BaseModel):
    model: str
    confidence: float

# A simple rule-based fallback if no ML model is available
# In a real scenario, this would use sentence-transformers to embed the prompt
# and classify it (e.g., simple chat vs coding vs reasoning).

@app.post("/v1/route", response_model=RouteResponse)
async def route_prompt(req: RouteRequest):
    text = req.prompt.lower()
    
    # Simple heuristic logic
    if any(keyword in text for keyword in ["code", "python", "java", "function"]):
        return RouteResponse(model="goktug-pro", confidence=0.85)
    elif any(keyword in text for keyword in ["math", "calculate", "solve"]):
        return RouteResponse(model="external-claude", confidence=0.90)
    else:
        # Default fallback
        return RouteResponse(model="goktug-mini", confidence=0.75)

@app.get("/health")
async def health():
    return {"status": "ok"}

import logging
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.vectorstore.chroma_store import get_collection_count
from src.retrieval.retriever import retrieve_player
from src.chatbot.chain import create_chat_chain, get_pick_evaluation

load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fantasy GodBot API",
    description="AI-powered fantasy football draft assistant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Per-session chat chains (keyed by session_id)
_chat_chains: dict = {}


# --- Request / Response models ---

class ChatRequest(BaseModel):
    message: str
    league_format: str = "redraft"
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    sources: list[str] = []


class PickEvaluationRequest(BaseModel):
    player_name: str
    pick_number: int
    league_format: str = "redraft"


class PickEvaluationResponse(BaseModel):
    grade: str
    analysis: str
    verdict: str
    raw_response: str


class HealthResponse(BaseModel):
    status: str
    players_indexed: int
    timestamp: str


# --- Auth dependency ---

def verify_refresh_key(x_api_key: str = Header(None)):
    expected = os.getenv("REFRESH_API_KEY", "")
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Returns service health and number of indexed players."""
    count = get_collection_count()
    return HealthResponse(
        status="ok",
        players_indexed=count,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Main chat endpoint. Maintains conversation history per session_id."""
    session_key = f"{request.session_id}:{request.league_format}"

    if session_key not in _chat_chains:
        logger.info(f"Creating new chat chain for session {session_key}")
        _chat_chains[session_key] = create_chat_chain(request.league_format)

    chain_fn = _chat_chains[session_key]

    try:
        response = chain_fn(request.message)
        return ChatResponse(response=response, sources=[])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal error generating response")


@app.post("/evaluate-pick", response_model=PickEvaluationResponse)
def evaluate_pick(request: PickEvaluationRequest):
    """Evaluates a specific draft pick with grade, analysis, and alternatives."""
    try:
        raw = get_pick_evaluation(
            player_name=request.player_name,
            pick_number=request.pick_number,
            league_format=request.league_format,
        )
        # Extract grade and verdict from the structured response
        grade = _extract_field(raw, "Pick Grade")
        verdict = _extract_field(raw, "Verdict")
        analysis = _extract_field(raw, "Analysis")

        return PickEvaluationResponse(
            grade=grade or "N/A",
            analysis=analysis or raw[:500],
            verdict=verdict or "",
            raw_response=raw,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Pick evaluation error: {e}")
        raise HTTPException(status_code=500, detail="Internal error evaluating pick")


@app.get("/player/{player_name}")
def get_player(player_name: str):
    """Returns the full document for a specific player."""
    doc = retrieve_player(player_name)
    if doc.startswith("No information found"):
        raise HTTPException(status_code=404, detail=doc)
    return {"player_name": player_name, "document": doc}


@app.post("/refresh")
def trigger_refresh(
    league_format: str = "redraft",
    force: bool = False,
    _: None = Depends(verify_refresh_key),
):
    """Triggers a full data refresh pipeline (requires API key)."""
    from src.ingestion.orchestrator import refresh_all
    try:
        logger.info(f"Triggering refresh: format={league_format}, force={force}")
        refresh_all(league_format=league_format, force=force)
        count = get_collection_count()
        return {"status": "ok", "players_indexed": count}
    except Exception as e:
        logger.error(f"Refresh error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _extract_field(text: str, field_name: str) -> str | None:
    """Extracts a bolded field value from the structured Claude response."""
    import re
    pattern = rf"\*\*{re.escape(field_name)}:\*\*\s*(.+?)(?=\n\*\*|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

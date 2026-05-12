"""
Tests for FastAPI endpoints using httpx test client.
"""
import os
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(scope="module")
def api_client():
    """Creates a FastAPI test client with heavy dependencies mocked."""
    with patch("src.vectorstore.chroma_store._get_embeddings") as mock_emb, \
         patch("langchain_chroma.Chroma") as mock_chroma:
        mock_emb.return_value = MagicMock()
        mock_chroma.return_value = MagicMock()

        from fastapi.testclient import TestClient
        from src.api.routes import app
        yield TestClient(app)


def test_health_ok(api_client):
    with patch("src.api.routes.get_collection_count", return_value=1844):
        response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["players_indexed"] == 1844


def test_health_returns_zero_when_empty(api_client):
    with patch("src.api.routes.get_collection_count", return_value=0):
        response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["players_indexed"] == 0


def test_chat_endpoint(api_client):
    mock_fn = MagicMock(return_value="CMC is an elite RB1 pick.")
    with patch("src.api.routes.create_chat_chain", return_value=mock_fn):
        response = api_client.post("/chat", json={
            "message": "Is CMC worth a first round pick?",
            "league_format": "redraft",
            "session_id": "test-session-fresh",
        })
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "sources" in data


def test_get_player_found(api_client):
    with patch("src.api.routes.retrieve_player",
               return_value="Christian McCaffrey | RB | SF ..."):
        response = api_client.get("/player/Christian%20McCaffrey")
    assert response.status_code == 200
    assert "document" in response.json()


def test_get_player_not_found(api_client):
    with patch("src.api.routes.retrieve_player",
               return_value="No information found for 'xyz'."):
        response = api_client.get("/player/xyz")
    assert response.status_code == 404


def test_refresh_requires_api_key(api_client):
    response = api_client.post("/refresh")
    assert response.status_code == 403


def test_refresh_with_valid_key(api_client):
    os.environ["REFRESH_API_KEY"] = "test-secret"
    with patch("src.ingestion.orchestrator.refresh_all"), \
         patch("src.api.routes.get_collection_count", return_value=1844):
        response = api_client.post(
            "/refresh",
            headers={"x-api-key": "test-secret"},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

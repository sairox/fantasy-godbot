"""
Tests for the RAG chain with mocked LLM and retriever.
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document


MOCK_PLAYER_DOC = Document(
    page_content=(
        "Christian McCaffrey | RB | San Francisco 49ers | Age: 28 | Experience: 8 years\n\n"
        "2026 DRAFT RANKINGS:\nStandard: RB1 (Overall: 2) | Half PPR: RB1 (Overall: 1)\n\n"
        "2025 PERFORMANCE:\nFantasy Points (Half PPR): 312.4 | Points Per Game: 19.5\n"
        "Games Played: 16 of 17 (missed 1) | Finish: RB1\n\n"
        "SIGNALS:\nTrend: Stable\nSleeper signal: No | Bust signal: No\n\n"
        "STATUS: Active | Depth Chart: Starter (1)"
    ),
    metadata={
        "player_id": "4046",
        "name": "Christian McCaffrey",
        "position": "RB",
        "team": "SF",
        "league_format": "redraft",
        "adp_2025": 1.2,
    },
)


@patch("src.chatbot.chain._get_llm")
@patch("src.chatbot.chain.get_retriever")
@patch("src.vectorstore.chroma_store.get_vectorstore")
def test_create_rag_chain_returns_response(mock_vs, mock_retriever_fn, mock_llm):
    from src.chatbot.chain import create_rag_chain

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [MOCK_PLAYER_DOC]
    mock_retriever_fn.return_value = mock_retriever

    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = MagicMock(content="**Pick Grade:** A\n**Verdict:** Draft CMC.")
    mock_llm.return_value = mock_llm_instance

    chain = create_rag_chain("redraft")
    assert chain is not None


@patch("src.chatbot.chain.retrieve_player")
@patch("src.chatbot.chain.retrieve_by_adp_range")
def test_get_pick_evaluation_calls_retrieve(mock_adp_range, mock_player):
    """Verifies that get_pick_evaluation calls the right retrieval functions."""
    mock_player.return_value = MOCK_PLAYER_DOC.page_content
    mock_adp_range.return_value = [MOCK_PLAYER_DOC]

    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import AIMessage as AI

    response_text = (
        "**Pick Grade:** A+\n"
        "**Analysis:** CMC is the best RB in fantasy.\n"
        "**Risk Factors:** Injury history.\n"
        "**Better Alternatives:** None.\n"
        "**Verdict:** Draft him at 1.01."
    )

    ai_msg = AI(content=response_text)
    with patch.object(ChatAnthropic, "invoke", return_value=ai_msg), \
         patch("src.chatbot.chain._get_llm", return_value=ChatAnthropic(
             model="claude-sonnet-4-5", anthropic_api_key="test-key"
         )):
        from src.chatbot.chain import get_pick_evaluation
        result = get_pick_evaluation("Christian McCaffrey", pick_number=3, league_format="redraft")

    mock_player.assert_called_once_with("Christian McCaffrey")
    mock_adp_range.assert_called_once_with(3, window=3)
    assert isinstance(result, str)
    assert "Pick Grade" in result


@patch("src.chatbot.chain._get_llm")
@patch("src.chatbot.chain.get_retriever")
@patch("src.vectorstore.chroma_store.get_vectorstore")
def test_create_chat_chain_returns_callable(mock_vs, mock_retriever_fn, mock_llm):
    from src.chatbot.chain import create_chat_chain

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [MOCK_PLAYER_DOC]
    mock_retriever_fn.return_value = mock_retriever

    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = MagicMock(content="Draft CMC.")
    mock_llm.return_value = mock_llm_instance

    chat_fn = create_chat_chain("redraft")
    assert callable(chat_fn)

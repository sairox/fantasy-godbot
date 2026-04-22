import logging
import os
from collections import deque
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

from src.retrieval.retriever import get_retriever, retrieve_player, retrieve_by_adp_range

load_dotenv()
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Fantasy GodBot, an expert fantasy football draft assistant.
You have access to 2026 expert rankings, 2025 season performance data, 2024 season data,
ADP history, and injury information for all fantasy-relevant NFL players.

The user is playing {league_format} fantasy football.

RESPONSE FORMAT:
When evaluating a draft pick, always structure your response as:

**Pick Grade:** [A+/A/A-/B+/B/B-/C+/C/D/F — use your judgment on format]
**Analysis:** [2-3 sentences backed by specific stats from the context]
**Risk Factors:** [injury history, age, situation concerns — only if relevant]
**Better Alternatives:** [0-3 players with better value near this ADP — only if they genuinely exist]
**Verdict:** [1 clear sentence — draft them or pass]

RULES:
- Always cite specific numbers from the context (fantasy points, ADP, finish rank)
- Never make up stats — only use what is in the retrieved context
- For alternatives, only suggest players within 3 picks of the user's current pick
- If a player has no better alternative, say so confidently
- Adjust advice based on league format: {league_format}
  - Redraft: prioritize 2025 performance, injury risk, current ADP
  - Dynasty: also consider age, years experience, long-term trajectory
- Be direct and opinionated — users want a clear recommendation, not a hedge
- If asked a general question (not about a specific player), answer conversationally
  using your knowledge of the players in context
"""


def _format_docs(docs) -> str:
    """Formats retrieved LangChain Documents into a single context string."""
    if not docs:
        return "No player data found in the knowledge base."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def _get_llm() -> ChatAnthropic:
    """Returns the Claude model instance."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise ValueError("ANTHROPIC_API_KEY is not set in .env")
    return ChatAnthropic(
        model="claude-sonnet-4-5",
        anthropic_api_key=api_key,
        max_tokens=1024,
    )


def create_rag_chain(league_format: str = "redraft"):
    """
    Creates the full LCEL RAG chain:
    user question → retriever → augmented prompt → Claude → response
    """
    retriever = get_retriever(league_format)
    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + "\n\nPLAYER CONTEXT:\n{context}"),
        ("human", "{question}"),
    ])

    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
            "league_format": RunnableLambda(lambda _: league_format),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def create_chat_chain(league_format: str = "redraft"):
    """
    Wraps the RAG chain with a sliding window of the last 10 exchanges.
    Returns a callable that accepts a question string and returns a response string.
    """
    retriever = get_retriever(league_format)
    llm = _get_llm()
    # Sliding window: each entry is (HumanMessage, AIMessage)
    history_window: deque = deque(maxlen=10)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + "\n\nPLAYER CONTEXT:\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    def run(question: str) -> str:
        # Flatten window into a list of messages
        flat_history = [msg for pair in history_window for msg in pair]

        docs = retriever.invoke(question)
        context = _format_docs(docs)

        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({
            "question": question,
            "context": context,
            "history": flat_history,
            "league_format": league_format,
        })

        history_window.append((HumanMessage(content=question), AIMessage(content=response)))
        return response

    return run


def get_pick_evaluation(player_name: str, pick_number: int,
                        league_format: str = "redraft") -> str:
    """
    Specialized function for draft pick evaluation.
    Retrieves the specific player + alternatives near the pick's ADP range.
    """
    llm = _get_llm()

    # Get the specific player's document
    player_context = retrieve_player(player_name)

    # Get alternative players near the pick number
    alternatives = retrieve_by_adp_range(pick_number, window=3)
    alternatives_text = _format_docs(alternatives)

    combined_context = (
        f"=== PLAYER BEING EVALUATED ===\n{player_context}\n\n"
        f"=== PLAYERS AVAILABLE NEAR PICK {pick_number} ===\n{alternatives_text}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",
         "The user is picking at pick {pick_number} in a {league_format} league.\n"
         "They are considering drafting {player_name}.\n\n"
         "Context:\n{context}\n\n"
         "Should they draft {player_name} at pick {pick_number}? "
         "Provide your pick grade, analysis, risk factors, better alternatives (only if they exist near this ADP), and verdict."),
    ])

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "pick_number": pick_number,
        "league_format": league_format,
        "player_name": player_name,
        "context": combined_context,
    })

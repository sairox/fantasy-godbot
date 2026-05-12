import logging
import os
import re
from collections import deque
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

from src.retrieval.retriever import (
    get_retriever, retrieve_player, retrieve_by_adp_range, retrieve_top_players,
)

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Nickname / alias map — detects player references in free-text questions
# ---------------------------------------------------------------------------

_NICKNAMES: dict[str, str] = {
    # RBs
    "cmc": "Christian McCaffrey",
    "mccaffrey": "Christian McCaffrey",
    "bijan": "Bijan Robinson",
    "gibbs": "Jahmyr Gibbs",
    "achane": "De'Von Achane",
    "barkley": "Saquon Barkley",
    "henry": "Derrick Henry",
    "breece": "Breece Hall",
    "taylor": "Jonathan Taylor",
    "kyren": "Kyren Williams",
    "pollard": "Tony Pollard",
    "mixon": "Joe Mixon",
    "stevenson": "Rhamondre Stevenson",
    "swift": "D'Andre Swift",
    # WRs
    "jefferson": "Justin Jefferson",
    "lamb": "CeeDee Lamb",
    "ceedee": "CeeDee Lamb",
    "chase": "Ja'Marr Chase",
    "jamarr": "Ja'Marr Chase",
    "hill": "Tyreek Hill",
    "tyreek": "Tyreek Hill",
    "nabers": "Malik Nabers",
    "marvin": "Marvin Harrison",
    "mhj": "Marvin Harrison",
    "puka": "Puka Nacua",
    "nacua": "Puka Nacua",
    "amon-ra": "Amon-Ra St. Brown",
    "evans": "Mike Evans",
    "diggs": "Stefon Diggs",
    "davante": "Davante Adams",
    "deebo": "Deebo Samuel",
    # TEs
    "kelce": "Travis Kelce",
    "bowers": "Brock Bowers",
    "andrews": "Mark Andrews",
    "laporte": "Sam LaPorta",
    "hockenson": "T.J. Hockenson",
    # QBs
    "mahomes": "Patrick Mahomes",
    "lamar": "Lamar Jackson",
    "hurts": "Jalen Hurts",
    "burrow": "Joe Burrow",
    "stroud": "C.J. Stroud",
    "purdy": "Brock Purdy",
    "love": "Jordan Love",
    "richardson": "Anthony Richardson",
    "herbert": "Justin Herbert",
    "tua": "Tua Tagovailoa",
    "prescott": "Dak Prescott",
    "dak": "Dak Prescott",
}

SYSTEM_PROMPT = (
    "You are Fantasy GodBot, an expert fantasy football draft assistant. "
    "You have access to 2026 expert rankings, 2025 season performance data, 2024 season data, "
    "ADP history, and advanced NGS stats (YPC, RYOE, targets, separation, CPOE, etc.) "
    "for all fantasy-relevant NFL players.\n\n"
    "The user is playing {league_format} fantasy football.\n\n"
    "RESPONSE FORMAT:\n"
    "When evaluating a draft pick, always structure your response as:\n\n"
    "**Pick Grade:** [A+/A/A-/B+/B/B-/C+/C/D/F]\n"
    "**Analysis:** [2-3 sentences backed by specific stats from the context]\n"
    "**Risk Factors:** [injury history, age, situation concerns - only if relevant]\n"
    "**Better Alternatives:** [0-3 players with better value near this ADP - only if they genuinely exist]\n"
    "**Verdict:** [1 clear sentence - draft them or pass]\n\n"
    "RULES:\n"
    "- Always cite specific numbers from the context (fantasy points, ADP, finish rank, NGS stats)\n"
    "- Never make up stats - only use what is in the retrieved context\n"
    "- When a player's full stats are available (rushing, receiving, passing), use ALL of them\n"
    "- For RBs, always mention both rushing AND receiving contribution\n"
    "- For QBs, mention rushing upside if relevant\n"
    "- For alternatives, only suggest players within 3 picks of the user's current pick\n"
    "- If a player has no better alternative, say so confidently\n"
    "- Adjust advice based on league format: {league_format}\n"
    "  - Redraft: prioritize 2025 performance, injury risk, current ADP\n"
    "  - Dynasty: also consider age, years experience, long-term trajectory\n"
    "- Be direct and opinionated - users want a clear recommendation, not a hedge\n"
    "- If asked a general question (not about a specific pick), answer conversationally "
    "using all player data in context"
)


def _parse_ranking_query(question: str) -> int | None:
    """Returns N if the question asks for a top-N list, else None."""
    q = question.lower()
    for pattern in [r"\btop[\s-]+(\d+)\b", r"\bbest\s+(\d+)\b",
                    r"\bfirst\s+(\d+)\s+picks?\b", r"\b(\d+)\s+best\b"]:
        m = re.search(pattern, q)
        if m:
            return min(int(m.group(1)), 30)

    first_round_triggers = [
        r"\bfirst[\s-]round\b", r"\b1st[\s-]round\b", r"\bround\s+1\b",
        r"\bdraft\s+order\b", r"\bdraft\s+board\b",
        r"\b1\.\d{2}\b",
        r"\bwho\s+(goes|are|will\s+go|would\s+go)\s+(first|top|early)",
        r"\btop\s+(overall|picks?|players?|guys?)\b",
        r"\bbest\s+(overall|players?|picks?)\s*(to\s+draft|available|this\s+year)?\b",
    ]
    for pattern in first_round_triggers:
        if re.search(pattern, q):
            return 12
    return None


def _find_player_in_question(question: str) -> str | None:
    """Detects a player nickname/name in the question and returns their full name."""
    q = question.lower()
    for nickname, full_name in _NICKNAMES.items():
        if re.search(r"\b" + re.escape(nickname) + r"\b", q):
            return full_name
    return None


def _format_docs(docs) -> str:
    """Formats retrieved LangChain Documents into a context string."""
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
    """LCEL RAG chain: question -> retriever -> Claude -> response."""
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
    Stateful chat chain with sliding-window history.
    Routes queries to rank-based or player-pinned retrieval as appropriate.
    Returns a callable: (question: str) -> str.
    """
    retriever = get_retriever(league_format)
    llm = _get_llm()
    history_window: deque = deque(maxlen=10)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + "\n\nPLAYER CONTEXT:\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    def run(question: str) -> str:
        flat_history = [msg for pair in history_window for msg in pair]

        top_n = _parse_ranking_query(question)
        if top_n:
            docs = retrieve_top_players(top_n, league_format)
            pinned = ""
        else:
            docs = retriever.invoke(question)
            pinned = ""
            # When the question names a specific player, pin their full doc first
            player_name = _find_player_in_question(question)
            if player_name:
                player_doc = retrieve_player(player_name)
                if not player_doc.startswith("No information found"):
                    pinned = f"=== {player_name.upper()} (DIRECT LOOKUP) ===\n{player_doc}\n\n"
                    logger.info("Pinned player doc for: %s", player_name)

        context = pinned + _format_docs(docs)
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
    """Evaluates a specific draft pick with grade, analysis, and alternatives."""
    llm = _get_llm()
    player_context = retrieve_player(player_name)
    alternatives = retrieve_by_adp_range(pick_number, window=3)

    combined_context = (
        f"=== PLAYER BEING EVALUATED ===\n{player_context}\n\n"
        f"=== PLAYERS AVAILABLE NEAR PICK {pick_number} ===\n{_format_docs(alternatives)}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",
         "The user is picking at pick {pick_number} in a {league_format} league.\n"
         "They are considering drafting {player_name}.\n\n"
         "Context:\n{context}\n\n"
         "Should they draft {player_name} at pick {pick_number}? "
         "Provide pick grade, analysis, risk factors, "
         "better alternatives (only if near this ADP), and verdict."),
    ])

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "pick_number": pick_number,
        "league_format": league_format,
        "player_name": player_name,
        "context": combined_context,
    })

import streamlit as st
import httpx
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Fantasy GodBot",
    page_icon="🏈",
    layout="wide",
)

# --- Session state init ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "league_format" not in st.session_state:
    st.session_state.league_format = None
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None


def _grade_color(grade: str) -> str:
    """Maps a letter grade to a CSS color."""
    grade = grade.strip().upper()
    if grade.startswith("A"):
        return "green"
    elif grade.startswith("B"):
        return "#d4a017"
    return "red"


def _send_message(message: str, league_format: str) -> str:
    """Calls the /chat endpoint and returns the response text."""
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{API_BASE}/chat",
                json={
                    "message": message,
                    "league_format": league_format,
                    "session_id": st.session_state.session_id,
                },
            )
            resp.raise_for_status()
            return resp.json()["response"]
    except httpx.ConnectError:
        return "❌ Cannot reach the Fantasy GodBot API. Make sure the server is running (`uv run uvicorn src.api.routes:app --reload`)."
    except Exception as e:
        return f"❌ Error: {e}"


def _trigger_refresh(league_format: str) -> str:
    """Calls /refresh endpoint."""
    api_key = os.getenv("REFRESH_API_KEY", "")
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(
                f"{API_BASE}/refresh",
                params={"league_format": league_format},
                headers={"x-api-key": api_key},
            )
            resp.raise_for_status()
            data = resp.json()
            return f"✅ Refreshed! {data.get('players_indexed', '?')} players indexed."
    except Exception as e:
        return f"❌ Refresh failed: {e}"


def _get_health() -> dict:
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{API_BASE}/health")
            return resp.json()
    except Exception:
        return {"status": "unreachable", "players_indexed": 0}


# --- Sidebar ---
with st.sidebar:
    st.title("🏈 Fantasy GodBot")
    st.markdown("---")

    # League format selector
    st.subheader("League Format")
    if st.session_state.league_format:
        new_format = st.selectbox(
            "Change format:",
            ["Redraft", "Half PPR", "Full PPR", "Dynasty"],
            index=["Redraft", "Half PPR", "Full PPR", "Dynasty"].index(
                {"redraft": "Redraft", "half_ppr": "Half PPR",
                 "ppr": "Full PPR", "dynasty": "Dynasty"}.get(
                    st.session_state.league_format, "Redraft"
                )
            ),
            label_visibility="collapsed",
        )
        format_map = {"Redraft": "redraft", "Half PPR": "half_ppr",
                      "Full PPR": "ppr", "Dynasty": "dynasty"}
        new_format_key = format_map[new_format]
        if new_format_key != st.session_state.league_format:
            st.session_state.league_format = new_format_key
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")

    # Health / data status
    health = _get_health()
    status_icon = "🟢" if health["status"] == "ok" else "🔴"
    st.caption(f"{status_icon} API: {health['status']}")
    st.caption(f"📊 Players indexed: {health.get('players_indexed', 0)}")
    if st.session_state.last_refresh:
        st.caption(f"🔄 Last refresh: {st.session_state.last_refresh}")

    st.markdown("---")

    # Refresh button
    if st.session_state.league_format and st.button("🔄 Refresh Data"):
        with st.spinner("Refreshing player data... (this takes a few minutes)"):
            msg = _trigger_refresh(st.session_state.league_format)
            st.session_state.last_refresh = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.success(msg)

    st.markdown("---")

    # Quick question buttons (only shown after format is selected)
    if st.session_state.league_format:
        st.subheader("Quick Questions")
        quick_questions = [
            "Who are the top 10 overall values for this draft?",
            "Who are the best RB sleepers this year?",
            "Best WR values in rounds 3-5?",
            "Which players are biggest busts to avoid?",
        ]
        for q in quick_questions:
            if st.button(q, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                with st.spinner("Thinking..."):
                    response = _send_message(q, st.session_state.league_format)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()


# --- Main content ---

# League format selection (first-time setup)
if not st.session_state.league_format:
    st.markdown(
        """
        <div style='text-align: center; padding: 60px 0 30px;'>
            <h1>Welcome to Fantasy GodBot 🏈</h1>
            <p style='font-size: 18px; color: #666;'>
                Your AI-powered 2026 fantasy football draft assistant.<br>
                Get pick grades, sleeper alerts, bust warnings, and expert-backed analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("What format are you playing?")
        format_choice = st.radio(
            "League format:",
            ["Redraft", "Half PPR", "Full PPR", "Dynasty"],
            label_visibility="collapsed",
            horizontal=False,
        )
        if st.button("Start Drafting →", use_container_width=True, type="primary"):
            format_map = {"Redraft": "redraft", "Half PPR": "half_ppr",
                          "Full PPR": "ppr", "Dynasty": "dynasty"}
            st.session_state.league_format = format_map[format_choice]
            st.rerun()
    st.stop()


# --- Chat interface ---
format_labels = {"redraft": "Redraft", "half_ppr": "Half PPR",
                 "ppr": "Full PPR", "dynasty": "Dynasty"}
current_label = format_labels.get(st.session_state.league_format, "Redraft")

st.title(f"Fantasy GodBot 🏈")
st.caption(f"League format: **{current_label}** | Ask about players, picks, or draft strategy")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        content = msg["content"]
        # Color-code pick grades in assistant responses
        if msg["role"] == "assistant" and "**Pick Grade:**" in content:
            import re
            grade_match = re.search(r"\*\*Pick Grade:\*\*\s*([A-F][+-]?)", content)
            if grade_match:
                grade = grade_match.group(1)
                color = _grade_color(grade)
                colored = content.replace(
                    f"**Pick Grade:** {grade}",
                    f"**Pick Grade:** <span style='color:{color}; font-size:1.2em; font-weight:bold'>{grade}</span>",
                    1,
                )
                st.markdown(colored, unsafe_allow_html=True)
            else:
                st.markdown(content)
        else:
            st.markdown(content)

# Chat input
if prompt := st.chat_input("Ask about a player, pick, or draft strategy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response = _send_message(prompt, st.session_state.league_format)

        # Color-code grade if present
        if "**Pick Grade:**" in response:
            import re
            grade_match = re.search(r"\*\*Pick Grade:\*\*\s*([A-F][+-]?)", response)
            if grade_match:
                grade = grade_match.group(1)
                color = _grade_color(grade)
                colored = response.replace(
                    f"**Pick Grade:** {grade}",
                    f"**Pick Grade:** <span style='color:{color}; font-size:1.2em; font-weight:bold'>{grade}</span>",
                    1,
                )
                st.markdown(colored, unsafe_allow_html=True)
            else:
                st.markdown(response)
        else:
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

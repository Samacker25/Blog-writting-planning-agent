from __future__ import annotations

import time
from datetime import date
from typing import Any, Dict, Iterator, Tuple

import streamlit as st

# Your backend
from backend import app


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Blog Writing Agent",
    layout="wide",
)


# -----------------------------
# 🎨 ChatGPT-like Styling
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    padding: 14px;
    border-radius: 14px;
    margin-bottom: 10px;
    max-width: 900px;
}

[data-testid="stChatMessage"][aria-label="user"] {
    background: rgba(99, 102, 241, 0.15);
}

[data-testid="stChatMessage"][aria-label="assistant"] {
    background: rgba(255,255,255,0.05);
}

/* Input */
textarea {
    border-radius: 12px !important;
}

/* Header */
.title {
    font-size: 42px;
    font-weight: 700;
    background: linear-gradient(90deg, #6366f1, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Title
# -----------------------------
st.markdown('<div class="title">🧠 Blog Writing Agent</div>', unsafe_allow_html=True)
st.caption("Generate high-quality blogs with AI (ChatGPT-style UI)")


# -----------------------------
# Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history_states" not in st.session_state:
    st.session_state.history_states = []


# -----------------------------
# Helpers
# -----------------------------
def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass

    try:
        for step in graph_app.stream(inputs, stream_mode="values"):
            yield ("values", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass

    out = graph_app.invoke(inputs)
    yield ("final", out)


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state


# -----------------------------
# Render Chat History
# -----------------------------
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show expandable agent state (optional)
        if msg["role"] == "assistant" and i < len(st.session_state.history_states):
            with st.expander("🔍 View agent steps"):
                st.json(st.session_state.history_states[i])


# -----------------------------
# Chat Input
# -----------------------------
user_input = st.chat_input("💡 Enter your blog topic...")

if user_input:
    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Show user instantly
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        thinking = st.empty()

        # Thinking animation
        thinking.markdown("🤔 Thinking...")
        time.sleep(0.4)
        thinking.markdown("🧠 Planning blog...")
        time.sleep(0.4)
        thinking.markdown("🔎 Researching...")
        time.sleep(0.4)

        streamed_text = ""
        current_state: Dict[str, Any] = {}

        inputs = {
            "topic": user_input,
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "as_of": date.today().isoformat(),
            "recency_days": 7,
            "sections": [],
            "merged_md": "",
            "md_with_placeholders": "",
            "image_specs": [],
            "final": "",
        }

        for kind, payload in try_stream(app, inputs):

            if kind in ("updates", "values"):
                current_state = extract_latest_state(current_state, payload)

                # Dynamic status updates
                if current_state.get("plan"):
                    thinking.markdown("🧠 Structuring content...")

                if current_state.get("evidence"):
                    thinking.markdown("🔎 Gathering sources...")

                if current_state.get("sections"):
                    thinking.markdown("✍️ Writing blog...")

            elif kind == "final":
                thinking.empty()

                final_md = payload.get("final", "")

                # Streaming effect
                for chunk in final_md.split(" "):
                    streamed_text += chunk + " "
                    placeholder.markdown(streamed_text + "▌")
                    time.sleep(0.01)

                placeholder.markdown(streamed_text)

                # Save assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_md
                })

                # Save state for debugging
                st.session_state.history_states.append(current_state)

                # Download button
                st.download_button(
                    "⬇️ Download Blog",
                    data=final_md,
                    file_name="blog.md",
                    mime="text/markdown"
                )
"""
app.py — Step 5: Streamlit web interface

What this file does:
  - Provides a chat interface in the browser
  - Accepts a natural language question
  - Calls rag_pipeline.py to get the answer
  - Displays the answer + expandable source cards

Run with:
  streamlit run src/app.py
"""

import sys
from pathlib import Path

# Make sure Python can find our other files
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from retriever import Retriever
from rag_pipeline import answer as get_answer

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SEBI MF Assistant",
    page_icon="📋",
    layout="wide",
)

# ── Load retriever once and cache it ─────────────────────────────────────────
# @st.cache_resource means this runs only once, even as the user keeps chatting
@st.cache_resource
def load_retriever():
    return Retriever()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📋 SEBI MF RegBot")

    st.markdown("**How it works:**")
    st.markdown("""
    1. You type a question
    2. System finds the most relevant chunks from SEBI documents
    3. phi3:mini reads those chunks and generates a grounded answer
    4. Sources are shown below the answer
    """)

    st.divider()

    st.markdown("**Try these questions:**")
    sample_questions = [
        "What is the maximum TER for large-cap mutual funds?",
        "How is the Risk-o-meter determined and reviewed?",
        "What are the eligibility criteria under MF Lite?",
        "What disclosures must distributors make to investors?",
        "How does swing pricing work and when is it triggered?",
        "What are the rules for categorization of hybrid schemes?",
    ]

    for q in sample_questions:
        # Clicking a sample question fills it into the chat input
        if st.button(q, use_container_width=True):
            st.session_state["prefill"] = q

    st.divider()
    st.caption("Documents: SEBI MF Circulars (2021–2026)")
    st.caption("LLM: phi3:mini via Ollama (local)")
    st.caption("Embeddings: BGE-Large (local)")


# ── Main page ─────────────────────────────────────────────────────────────────
st.title("🏛️ SEBI Mutual Fund Regulatory Assistant")
st.markdown(
    "Ask any question about SEBI Mutual Fund regulations. "
    "All answers are grounded in official SEBI documents."
)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 View Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"**{src['label']}** — `{src['file']}` | Page **{src['page']}**")
                    st.caption(src["snippet"])
                    st.divider()

# Handle prefilled question from sidebar button
prefill = st.session_state.pop("prefill", "")
query   = st.chat_input("Ask about SEBI MF regulations...") or prefill

# ── Handle new query ──────────────────────────────────────────────────────────
if query:
    # Show the user's question
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate and show the answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            retriever = load_retriever()
            result    = get_answer(query, retriever)

        # Display answer
        st.markdown(result["answer"])

        # Display source cards
        if result["sources"]:
            with st.expander("📄 View Sources", expanded=True):
                # Show sources in columns (up to 3 per row)
                num_cols = min(len(result["sources"]), 3)
                cols     = st.columns(num_cols)

                for i, src in enumerate(result["sources"]):
                    with cols[i % num_cols]:
                        st.markdown(f"**{src['label']}**")
                        st.markdown(f"📁 `{src['file']}`")
                        st.markdown(f"📄 Page **{src['page']}**")
                        st.caption(src["snippet"])

    # Save assistant message to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })

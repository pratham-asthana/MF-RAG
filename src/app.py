import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from retriever import Retriever
from rag_pipeline import answer as get_answer

st.set_page_config(
    page_title="SEBI MF Assistant",
    layout="wide",
)

@st.cache_resource
def load_retriever():
    return Retriever()


st.title("SEBI Mutual Fund Regulatory Assistant")
st.markdown(
    "Ask any question about SEBI Mutual Fund regulations. "
    "All answers are grounded in official SEBI documents."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(" View Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"**{src['label']}** — `{src['file']}` | Page **{src['page']}**")
                    st.caption(src["snippet"])
                    st.divider()

prefill = st.session_state.pop("prefill", "")
query   = st.chat_input("Ask about SEBI MF regulations...") or prefill

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            retriever = load_retriever()
            result    = get_answer(query, retriever)

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("View Sources", expanded=True):
                num_cols = min(len(result["sources"]), 3)
                cols     = st.columns(num_cols)

                for i, src in enumerate(result["sources"]):
                    with cols[i % num_cols]:
                        st.markdown(f"**{src['label']}**")
                        st.markdown(f" `{src['file']}`")
                        st.markdown(f" Page **{src['page']}**")
                        st.caption(src["snippet"])

    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })

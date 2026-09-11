import os
import streamlit as st
from rag_chain import answer_question, _vectorstore

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="College Admission Agent",
    page_icon="🎓",
    layout="centered",
)

# ── Sidebar: loaded documents ─────────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 College Admission Agent")
    st.markdown("---")
    st.subheader("📂 Loaded Documents")

    # Collect unique source filenames stored in ChromaDB metadata
    try:
        all_metadata = _vectorstore.get(include=["metadatas"])["metadatas"]
        doc_names = sorted({
            os.path.basename(m.get("source", "unknown"))
            for m in all_metadata
            if m
        })
        if doc_names:
            for name in doc_names:
                st.markdown(f"- `{name}`")
        else:
            st.caption("No documents found in the knowledge base.")
    except Exception:
        st.caption("Could not load document list.")

    st.markdown("---")
    st.caption("Powered by IBM Granite · ChromaDB · LangChain")

# ── Session state: chat history ───────────────────────────────────────────────
# Each entry: {"role": "user"|"assistant", "content": str, "sources": list[str]|None}
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Main header ───────────────────────────────────────────────────────────────
st.title("🎓 College Admission Agent")
st.caption("Ask me anything about admissions, eligibility, fees, or course selection.")

# ── Render existing chat history ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"- `{src}`")

# ── Chat input ────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Type your question here..."):

    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": user_input, "sources": None})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = answer_question(user_input)

        answer  = result["answer"]
        sources = result["sources"]

        st.markdown(answer)

        if sources:
            with st.expander("📎 Sources", expanded=False):
                for src in sources:
                    st.markdown(f"- `{src}`")

    # Persist assistant message to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
    })

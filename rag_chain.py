import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── Embedding model (must match the one used in ingest.py) ────────────────────
_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ── Load persistent ChromaDB ──────────────────────────────────────────────────
_vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=_embeddings,
)

# ── Retriever: return top-4 most relevant chunks ──────────────────────────────
_retriever = _vectorstore.as_retriever(search_kwargs={"k": 4})

# ── IBM Granite via Ollama ─────────────────────────────────────────────────────
_llm = ChatOllama(model="granite3.2:2b", temperature=0)

# ── Prompt ────────────────────────────────────────────────────────────────────
_SYSTEM = (
    "You are a helpful college admissions assistant. "
    "Answer the user's question using ONLY the information provided in the context below. "
    "If the answer is not present in the context, say \"I don't have enough information to answer that.\" "
    "At the end of your answer, cite the source filename(s) you used on a new line prefixed with 'Sources:'."
)

_HUMAN = (
    "Context:\n{context}\n\n"
    "Question: {question}"
)

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", _HUMAN),
])

# ── Helper: format retrieved docs into a single context string ────────────────
def _format_docs(docs):
    return "\n\n".join(
        f"[{os.path.basename(doc.metadata.get('source', 'unknown'))}]\n{doc.page_content}"
        for doc in docs
    )

# ── RAG chain ─────────────────────────────────────────────────────────────────
_chain = (
    {
        "context":  _retriever | _format_docs,
        "question": RunnablePassthrough(),
    }
    | _prompt
    | _llm
    | StrOutputParser()
)


def answer_question(query: str) -> dict:
    """
    Run a RAG query against the ChromaDB knowledge base.

    Parameters
    ----------
    query : str
        The question to answer.

    Returns
    -------
    dict
        {
            "answer":  str,          # full model response including inline sources
            "sources": list[str],    # deduplicated list of source filenames
        }
    """
    # Retrieve source documents separately so we can surface them cleanly
    retrieved_docs = _retriever.invoke(query)
    sources = sorted({
        os.path.basename(doc.metadata.get("source", "unknown"))
        for doc in retrieved_docs
    })

    # Run the full chain to get the grounded answer
    answer = _chain.invoke(query)

    return {"answer": answer, "sources": sources}


# ── Quick smoke-test when run directly ────────────────────────────────────────
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are the admission requirements?"
    print(f"Question: {query}\n")
    result = answer_question(query)
    print(result["answer"])
    print(f"\nSources: {', '.join(result['sources'])}")

"""Streamlit web app for the MG ZS Car Manual RAG project.

Supports two LLM providers:
  * OpenRouter (existing pipeline)
  * Google Gemini

API keys are entered in the UI; nothing is read from .env.
Reuses car_rag's ingestion, retrieval, prompting, and answering logic.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import streamlit as st
from chromadb import PersistentClient

from car_rag import rag
from car_rag.rerank import OpenRouterReranker

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

st.set_page_config(
    page_title="MG ZS Car Manual Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "messages": [],
    "provider": "openrouter",  # "openrouter" | "gemini"
    "api_key": "",
    "openrouter_base_url": "https://openrouter.ai/api/v1",
    "openrouter_chat_model": "openai/gpt-4o-mini",
    "openrouter_embed_model": "openai/text-embedding-3-small",
    "gemini_chat_model": "gemini-3.6-flash",
    "gemini_embed_model": "models/gemini-embedding-001",
    "index_ready": None,
    "_last_provider": None,
    "_cache_buster": 0,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# When provider or API key changes, invalidate cached resources by bumping
# _cache_buster and clearing stale widget values so new `value=` takes effect.
if st.session_state._last_provider is not None and \
   st.session_state._last_provider != st.session_state.provider:
    _old = st.session_state._last_provider
    _new = st.session_state.provider
    for _key in ("gemini_chat_model", "gemini_embed_model",
                 "openrouter_chat_model", "openrouter_embed_model"):
        if _key in st.session_state:
            del st.session_state[_key]
    st.session_state._cache_buster += 1
st.session_state._last_provider = st.session_state.provider

# Detect API key change
if "_last_api_key" not in st.session_state:
    st.session_state._last_api_key = ""
if st.session_state._last_api_key != st.session_state.api_key:
    st.session_state._cache_buster += 1
st.session_state._last_api_key = st.session_state.api_key


# ---------------------------------------------------------------------------
# App settings (purely from UI, no .env)
# ---------------------------------------------------------------------------
class AppSettings:
    """Simple namespace holding all tunables from the UI. No .env dependency."""

    def __init__(self, provider: str, api_key: str) -> None:
        self.provider = provider
        self.api_key = api_key.strip()
        self.openrouter_base_url = st.session_state.get(
            "openrouter_base_url", "https://openrouter.ai/api/v1"
        )
        self.openrouter_chat_model = st.session_state.get(
            "openrouter_chat_model", "openai/gpt-4o-mini"
        )
        self.openrouter_embed_model = st.session_state.get(
            "openrouter_embed_model", "openai/text-embedding-3-small"
        )
        self.gemini_chat_model = st.session_state.get(
            "gemini_chat_model", "gemini-3.8-flash"
        )
        self.gemini_embed_model = st.session_state.get(
            "gemini_embed_model", "models/gemini-embedding-001"
        )
        self.answer_max_tokens = 512
        self.retriever_k = 5
        self.hybrid_bm25_weight = 0.5
        self.rerank_top_n = 3
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.data_path = "data/mg-zs-warning-messages.html"
        self.chroma_dir = "chroma_dir" if False else "/tmp/chroma_db"


def api_key_label() -> str:
    return "Google Gemini API Key" if st.session_state.provider == "gemini" else "OpenRouter API Key"


def api_key_placeholder() -> str:
    return "AIza..." if st.session_state.provider == "gemini" else "sk-or-v1-..."


# ---------------------------------------------------------------------------
# Resource builders (cached, invalidated by _cache_buster)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_embeddings(_cache_buster: int, provider: str, api_key: str,
                     base_url: str,
                     openrouter_embed_model: str, gemini_embed_model: str):
    if provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model=gemini_embed_model,
            google_api_key=api_key,
        )
    return OpenAIEmbeddings(
        model=openrouter_embed_model,
        api_key=api_key,
        base_url=base_url,
    )


@st.cache_resource(show_spinner=False)
def build_llm(_cache_buster: int, provider: str, api_key: str,
              base_url: str,
              openrouter_chat_model: str, gemini_chat_model: str, max_tokens: int):
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=gemini_chat_model,
            google_api_key=api_key,
            temperature=0,
            max_output_tokens=max_tokens,
        )
    return ChatOpenAI(
        model=openrouter_chat_model,
        temperature=0,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
    )


@st.cache_resource(show_spinner=False)
def build_reranker_cached(_cache_buster: int, api_key: str, base_url: str,
                          model: str, top_n: int):
    return OpenRouterReranker(
        api_key=api_key,
        base_url=base_url,
        model=model,
        top_n=top_n,
    )


def load_vectorstore_direct(embeddings, chroma_dir: str):
    return rag.load_vectorstore(embeddings, chroma_dir)


def build_retriever_direct(
    mode: str,
    vectorstore,
    chunks,
    k: int,
    bm25_weight: float,
):
    return rag.build_retriever(
        mode, vectorstore, chunks, k, bm25_weight
    )


def check_index_ready(embeddings, chroma_dir: str) -> bool:
    try:
        vs = rag.load_vectorstore(embeddings, chroma_dir)
        return vs._collection.count() > 0
    except Exception:
        return False


def run_ingest(rebuild: bool, provider: str, api_key: str, base_url: str,
               openrouter_embed_model: str, gemini_embed_model: str,
               chunk_size: int, chunk_overlap: int, data_path: str,
               chroma_dir: str) -> tuple[bool, str]:
    """Build the Chroma index using provider-aware embeddings (no OpenRouter key needed for Gemini)."""
    try:
        if rebuild and Path(chroma_dir).exists():
            # ChromaDB holds file locks on its .bin files on Windows. Use Chroma's
            # own reset() to release handles first, then fall back to rmtree with retries.
            try:
                if hasattr(PersistentClient, "reset"):
                    client = PersistentClient(path=chroma_dir)
                    client.reset()
                    st.toast(f"Reset existing index at '{chroma_dir}'.", icon="🗑️")
                else:
                    raise AttributeError("PersistentClient.reset not available")
            except Exception:
                removed = False
                for _ in range(5):
                    try:
                        shutil.rmtree(chroma_dir)
                        removed = True
                        break
                    except PermissionError:
                        time.sleep(1)
                if not removed:
                    st.toast(
                        f"Could not remove '{chroma_dir}' (file in use). "
                        "Close other apps using it and try again.",
                        icon="⚠️",
                    )
                else:
                    st.toast(f"Removed existing index at '{chroma_dir}'.", icon="🗑️")

        docs = rag.load_documents(data_path)
        chunks = rag.split_documents(docs, chunk_size, chunk_overlap)

        # Provider-aware embeddings: never call rag.make_embeddings() here because
        # it requires the OpenRouter API key even when using Gemini.
        if provider == "gemini":
            embeddings = GoogleGenerativeAIEmbeddings(
                model=gemini_embed_model,
                google_api_key=api_key,
            )
        else:
            embeddings = OpenAIEmbeddings(
                model=openrouter_embed_model,
                api_key=api_key,
                base_url=base_url,
            )

        rag.build_vectorstore(chunks, embeddings, chroma_dir)
        st.session_state.index_ready = True
        return True, f"Indexed {len(chunks)} chunks from '{data_path}'."
    except Exception as exc:  # noqa: BLE001
        st.session_state.index_ready = False
        return False, f"Ingest failed: {exc}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Provider")
    provider = st.radio(
        "LLM provider",
        options=["OpenRouter", "Google Gemini"],
        index=0 if st.session_state.provider == "openrouter" else 1,
        horizontal=True,
    )
    st.session_state.provider = "openrouter" if provider == "OpenRouter" else "gemini"

    st.subheader("API Key")
    st.text_input(
        api_key_label(),
        type="password",
        key="api_key",
        placeholder=api_key_placeholder(),
        help="Your API key for the selected provider. Stored only in this browser session.",
    )
    key_present = bool(st.session_state.api_key.strip())

    st.divider()

    st.subheader("Models")
    if st.session_state.provider == "openrouter":
        st.text_input("Base URL", key="openrouter_base_url", value="https://openrouter.ai/api/v1")
        st.text_input("Chat model", key="openrouter_chat_model", value="openai/gpt-4o-mini")
        st.text_input("Embedding model", key="openrouter_embed_model", value="openai/text-embedding-3-small")
        st.caption("Reranker: cohere/rerank-v3.5 (OpenRouter only)")
    else:
        st.text_input("Chat model", key="gemini_chat_model", value="gemini-3.6-flash")
        st.text_input("Embedding model", key="gemini_embed_model", value="models/gemini-embedding-001")
        st.caption("Reranking is unavailable for Gemini (OpenRouter-only feature).")

    st.divider()

    st.subheader("Index")
    settings = AppSettings(st.session_state.provider, st.session_state.api_key)
    index_dir = Path(settings.chroma_dir)
    st.caption(f"Index directory: `{index_dir}`")
    st.caption(
        "⚠️ The index is provider-specific. If you switch providers, "
        "click **Ingest / Rebuild Index** to rebuild with the new provider's embeddings."
    )

    if st.button("Ingest / Rebuild Index", use_container_width=True, type="primary"):
        if not key_present:
            st.error("Enter an API key before ingesting.")
        else:
            with st.spinner("Ingesting manual... this may take a minute."):
                ok, msg = run_ingest(
                    rebuild=True,
                    provider=st.session_state.provider,
                    api_key=st.session_state.api_key.strip(),
                    base_url=st.session_state.get("openrouter_base_url", "https://openrouter.ai/api/v1"),
                    openrouter_embed_model=st.session_state.get(
                        "openrouter_embed_model", "openai/text-embedding-3-small"
                    ),
                    gemini_embed_model=st.session_state.get(
                        "gemini_embed_model", "models/gemini-embedding-001"
                    ),
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                    data_path=settings.data_path,
                    chroma_dir=settings.chroma_dir,
                )
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    if st.session_state.index_ready is True:
        st.success("Index is ready.")
    elif st.session_state.index_ready is False:
        st.warning("Index not ready. Run Ingest first.")

    st.divider()

    st.subheader("Retrieval")
    retriever_mode = st.selectbox(
        "Retriever",
        options=["dense", "hybrid"],
        index=0,
        help="dense = vector search only; hybrid = BM25 + vector ensemble.",
    )
    use_rerank = st.checkbox(
        "Apply reranking",
        value=False,
        disabled=(st.session_state.provider == "gemini"),
        help="Re-rank retrieved chunks with OpenRouter /rerank (OpenRouter only).",
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("🚗 MG ZS Warning Message Assistant")
st.caption(
    "Ask about MG ZS dashboard warning messages. Answers are grounded in the "
    "MG ZS warning-messages manual via retrieval-augmented generation (RAG)."
)

# Validate prerequisites
if not key_present:
    st.error(f"Please enter your {api_key_label()} in the sidebar before asking questions.")
    st.stop()

api_key = st.session_state.api_key.strip()
settings = AppSettings(st.session_state.provider, api_key)

# Build cached resources
try:
    embeddings = build_embeddings(
        st.session_state._cache_buster,
        st.session_state.provider,
        api_key,
        st.session_state.get("openrouter_base_url", "https://openrouter.ai/api/v1"),
        st.session_state.get("openrouter_embed_model", "openai/text-embedding-3-small"),
        st.session_state.get("gemini_embed_model", "models/gemini-embedding-001"),
    )

    llm = build_llm(
        st.session_state._cache_buster,
        st.session_state.provider,
        api_key,
        st.session_state.get("openrouter_base_url", "https://openrouter.ai/api/v1"),
        st.session_state.get("openrouter_chat_model", "openai/gpt-4o-mini"),
        st.session_state.get("gemini_chat_model", "gemini-3.8-flash"),
        settings.answer_max_tokens,
    )

    index_ready = check_index_ready(embeddings, settings.chroma_dir)
    st.session_state.index_ready = index_ready

    if not index_ready:
        st.info(
            "No search index found yet. Click **Ingest / Rebuild Index** in the "
            "sidebar to build it from the manual, then ask your question."
        )
        st.stop()

    chunks_for_hybrid = None
    if retriever_mode == "hybrid":
        chunks_for_hybrid = rag.split_documents(
            rag.load_documents(settings.data_path),
            settings.chunk_size,
            settings.chunk_overlap,
        )

    vectorstore = load_vectorstore_direct(embeddings, settings.chroma_dir)
    retriever = build_retriever_direct(
        retriever_mode,
        vectorstore,
        chunks_for_hybrid,
        settings.retriever_k,
        settings.hybrid_bm25_weight,
    )

    reranker = None
    if use_rerank and st.session_state.provider == "openrouter":
        reranker = build_reranker_cached(
            st.session_state._cache_buster,
            api_key,
            st.session_state.get("openrouter_base_url", "https://openrouter.ai/api/v1"),
            "cohere/rerank-v3.5",
            settings.rerank_top_n,
        )

except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to initialize RAG pipeline: {exc}")
    st.stop()

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("contexts"):
            with st.expander(
                f"Sources ({len(message['contexts'])} chunk(s))",
                expanded=False,
            ):
                for i, ctx in enumerate(message["contexts"], 1):
                    scores = message.get("rerank_scores") or [None] * len(message["contexts"])
                    score = scores[i - 1] if i - 1 < len(scores) else None
                    score_txt = f" (rerank score: {score:.4f})" if score is not None else ""
                    st.markdown(f"**Chunk {i}**{score_txt}")
                    st.write(ctx[:1000] + ("…" if len(ctx) > 1000 else ""))
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask about a warning message (e.g. 'What does the ABS Fault mean?')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking…"):
                result = rag.answer(prompt, retriever, llm, reranker)

            response = result["response"]
            contexts = result.get("contexts", [])

            # Collect rerank scores from the docs already retrieved inside rag.answer().
            # We re-run retrieve() only when reranking is on, to surface scores without
            # changing the answer (the answer used the same underlying retrieval).
            rerank_scores = None
            if use_rerank and reranker is not None:
                try:
                    docs = rag.retrieve(retriever, prompt, reranker)
                    rerank_scores = [
                        float(d.metadata.get("rerank_score", 0.0)) for d in docs
                    ]
                except Exception:
                    rerank_scores = None

            st.markdown(response)

            if contexts:
                with st.expander(
                    f"Sources ({len(contexts)} chunk(s))",
                    expanded=False,
                ):
                    for i, ctx in enumerate(contexts, 1):
                        score = (
                            rerank_scores[i - 1] if rerank_scores else None
                        )
                        score_txt = f" (rerank score: {score:.4f})" if score is not None else ""
                        st.markdown(f"**Chunk {i}**{score_txt}")
                        st.write(ctx[:1000] + ("…" if len(ctx) > 1000 else ""))
                        st.divider()

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                    "contexts": contexts,
                    "rerank_scores": rerank_scores,
                }
            )

        except Exception as exc:  # noqa: BLE001
            st.error(f"Error generating answer: {exc}")

if st.session_state.messages:
    if st.button("Clear chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()

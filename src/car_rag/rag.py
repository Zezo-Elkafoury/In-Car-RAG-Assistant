"""Core RAG pipeline: ingestion, indexing, retrieval, answering."""

from pathlib import Path
from typing import Literal

from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from car_rag.config import Settings
from car_rag.prompts import build_prompt
from car_rag.rerank import OpenRouterReranker

RetrieverMode = Literal["dense", "hybrid"]


def make_chat_llm(
    settings: Settings, model: str | None = None, max_tokens: int | None = None
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or settings.openrouter_chat_model,
        temperature=0,
        api_key=settings.require_api_key(),
        base_url=settings.openrouter_base_url,
        max_tokens=max_tokens if max_tokens is not None else settings.answer_max_tokens,
    )


def make_embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.openrouter_embedding_model,
        api_key=settings.require_api_key(),
        base_url=settings.openrouter_base_url,
    )


def load_documents(data_path: str | Path) -> list[Document]:
    loader = UnstructuredHTMLLoader(file_path=str(data_path))
    return loader.load()


def split_documents(
    docs: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def build_vectorstore(
    chunks: list[Document], embeddings: OpenAIEmbeddings, persist_directory: str | Path
) -> Chroma:
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory),
    )


def load_vectorstore(
    embeddings: OpenAIEmbeddings, persist_directory: str | Path
) -> Chroma:
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    if vectorstore._collection.count() == 0:
        raise RuntimeError(
            f"No indexed documents in '{persist_directory}'. Run `car-rag ingest` first."
        )
    return vectorstore


def build_retriever(
    mode: RetrieverMode,
    vectorstore: Chroma,
    chunks: list[Document] | None,
    k: int,
    bm25_weight: float = 0.5,
) -> BaseRetriever:
    """Dense (vector-only) or hybrid (BM25 + vector via RRF ensemble) retriever."""
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    if mode == "dense":
        return vector_retriever
    if chunks is None:
        raise ValueError("Hybrid retrieval needs the split chunks for BM25. Re-ingest first.")
    bm25 = BM25Retriever.from_documents(chunks, k=k)
    return EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=[bm25_weight, 1.0 - bm25_weight],
    )


def retrieve(
    retriever: BaseRetriever,
    query: str,
    reranker: OpenRouterReranker | None = None,
) -> list[Document]:
    docs = retriever.invoke(query)
    if reranker is not None:
        docs = reranker.compress_documents(docs, query)
    return docs


def answer(
    question: str,
    retriever: BaseRetriever,
    llm: ChatOpenAI,
    reranker: OpenRouterReranker | None = None,
) -> dict:
    """Run retrieval (+ optional rerank) and generate an answer.

    Returns {"response": str, "contexts": list[str]} - contexts are the raw
    retrieved chunk texts, needed for RAGAS evaluation.
    """
    docs = retrieve(retriever, question, reranker)
    context = "\n\n".join(doc.page_content for doc in docs)
    chain = build_prompt() | llm | StrOutputParser()
    response = chain.invoke({"context": context, "question": question})
    return {"response": response, "contexts": [doc.page_content for doc in docs]}

"""CLI: car-rag ingest | ask | eval"""

import shutil
from pathlib import Path
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv

load_dotenv()

from car_rag import rag  # noqa: E402
from car_rag.config import get_settings  # noqa: E402
from car_rag.eval import REPORT_COLUMNS, run_evaluation  # noqa: E402
from car_rag.rerank import OpenRouterReranker  # noqa: E402

app = typer.Typer(help="Car-manual RAG over MG ZS warning messages (OpenRouter).")


@app.command()
def ingest(
    rebuild: Annotated[bool, typer.Option(help="Delete the existing index first.")] = False,
) -> None:
    """Load the manual HTML, chunk it, and build the Chroma index."""
    settings = get_settings()
    settings.require_api_key()
    if rebuild and Path(settings.chroma_dir).exists():
        shutil.rmtree(settings.chroma_dir)
        print(f"Removed existing index at '{settings.chroma_dir}'.")
    docs = rag.load_documents(settings.data_path)
    chunks = rag.split_documents(docs, settings.chunk_size, settings.chunk_overlap)
    embeddings = rag.make_embeddings(settings)
    rag.build_vectorstore(chunks, embeddings, settings.chroma_dir)
    print(f"Indexed {len(chunks)} chunks from '{settings.data_path}'.")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Driver's question.")],
    retriever: Annotated[str, typer.Option(help="dense or hybrid.")] = "dense",
    rerank: Annotated[bool, typer.Option(help="Apply OpenRouter reranking.")] = False,
) -> None:
    """Answer a single question with the RAG pipeline."""
    settings = get_settings()
    settings.require_api_key()
    embeddings = rag.make_embeddings(settings)
    llm = rag.make_chat_llm(settings)
    chunks = (
        rag.split_documents(
            rag.load_documents(settings.data_path),
            settings.chunk_size,
            settings.chunk_overlap,
        )
        if retriever == "hybrid"
        else None
    )
    vectorstore = rag.load_vectorstore(embeddings, settings.chroma_dir)
    base = rag.build_retriever(
        retriever, vectorstore, chunks, settings.retriever_k, settings.hybrid_bm25_weight
    )
    reranker = (
        OpenRouterReranker(
            api_key=settings.require_api_key(),
            base_url=settings.openrouter_base_url,
            model=settings.rerank_model,
            top_n=settings.rerank_top_n,
        )
        if rerank
        else None
    )
    result = rag.answer(question, base, llm, reranker)
    print(f"\nQ: {question}\n\nA: {result['response']}\n")
    print(f"Sources used ({len(result['contexts'])} chunks).")


@app.command(name="eval")
def eval_cmd(
    dataset: Annotated[str, typer.Option(help="Path to .jsonl eval dataset.")] = (
        "evals/datasets/combined.jsonl"
    ),
    retriever: Annotated[str, typer.Option(help="dense or hybrid.")] = "dense",
    rerank: Annotated[bool, typer.Option(help="Apply OpenRouter reranking.")] = False,
    limit: Annotated[Optional[int], typer.Option(help="Score only the first N rows.")] = None,
    label: Annotated[str, typer.Option(help="Run label for the report filename.")] = "run",
) -> None:
    """Score the RAG pipeline with RAGAS and write a JSON + Markdown report."""
    report = run_evaluation(dataset, retriever, rerank, limit, label)
    print("\nAggregate scores:")
    for col in REPORT_COLUMNS:
        print(f"  {col}: {report['aggregate'][col]:.3f}")


if __name__ == "__main__":
    app()

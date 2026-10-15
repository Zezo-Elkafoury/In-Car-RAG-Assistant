"""RAGAS evaluation harness: AnswerAccuracy + Faithfulness + Context Precision."""

import json
from datetime import datetime, timezone
from pathlib import Path

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerAccuracy, Faithfulness, LLMContextPrecisionWithReference
from ragas.run_config import RunConfig

from car_rag import rag
from car_rag.config import Settings
from car_rag.rerank import OpenRouterReranker

REPORT_COLUMNS = ["nv_accuracy", "faithfulness", "llm_context_precision_with_reference"]


def load_eval_rows(dataset_path: str | Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def build_judge(settings: Settings) -> LangchainLLMWrapper:
    judge_llm = rag.make_chat_llm(
        settings, model=settings.eval_llm_model, max_tokens=settings.judge_max_tokens
    )
    return LangchainLLMWrapper(judge_llm)


def build_metrics(judge: LangchainLLMWrapper) -> list:
    return [
        AnswerAccuracy(llm=judge),  # Output Accuracy: string evaluation vs reference
        Faithfulness(llm=judge),  # Grounded in retrieved context (0-1)
        LLMContextPrecisionWithReference(llm=judge),  # Retrieval relevance (0-1)
    ]


def run_evaluation(
    dataset_path: str | Path,
    retriever_mode: rag.RetrieverMode = "dense",
    use_rerank: bool = False,
    limit: int | None = None,
    label: str = "run",
    reports_dir: str | Path = "evals/reports",
) -> dict:
    """Run the RAG pipeline over the dataset and score it with RAGAS."""
    settings = Settings()
    settings.require_api_key()

    rows = load_eval_rows(dataset_path, limit)
    if not rows:
        raise ValueError(f"No evaluation rows found in '{dataset_path}'.")

    embeddings = rag.make_embeddings(settings)
    llm = rag.make_chat_llm(settings)
    chunks: list | None = None
    if retriever_mode == "hybrid":
        chunks = rag.split_documents(
            rag.load_documents(settings.data_path),
            settings.chunk_size,
            settings.chunk_overlap,
        )
    vectorstore = rag.load_vectorstore(embeddings, settings.chroma_dir)
    retriever = rag.build_retriever(
        retriever_mode, vectorstore, chunks, settings.retriever_k,
        settings.hybrid_bm25_weight,
    )
    reranker = (
        OpenRouterReranker(
            api_key=settings.require_api_key(),
            base_url=settings.openrouter_base_url,
            model=settings.rerank_model,
            top_n=settings.rerank_top_n,
        )
        if use_rerank
        else None
    )

    samples: list[SingleTurnSample] = []
    failed: list[dict] = []
    for i, row in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {row['user_input'][:80]}...")
        try:
            result = rag.answer(row["user_input"], retriever, llm, reranker)
        except Exception as exc:  # noqa: BLE001 - one bad sample must not kill the run
            print(f"  FAILED: {exc}")
            failed.append({"user_input": row["user_input"], "error": str(exc)[:300]})
            continue
        samples.append(
            SingleTurnSample(
                user_input=row["user_input"],
                response=result["response"],
                reference=row["reference"],
                retrieved_contexts=result["contexts"],
            )
        )
    if not samples:
        raise RuntimeError("All samples failed. Aborting evaluation.")

    judge = build_judge(settings)
    eval_result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=build_metrics(judge),
        llm=judge,
        # Serial judge calls: OpenRouter free-tier keys reject concurrent
        # requests once the in-flight budget is exhausted (HTTP 402).
        run_config=RunConfig(max_workers=1, max_retries=5),
    )
    df = eval_result.to_pandas()

    per_sample = [
        {
            "user_input": s.user_input,
            "reference": s.reference,
            "response": s.response,
            **{col: float(df.loc[idx, col]) for col in REPORT_COLUMNS},
        }
        for idx, s in enumerate(samples)
    ]
    aggregate = {col: float(df[col].mean()) for col in REPORT_COLUMNS}
    report = {
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "retriever": retriever_mode,
        "rerank": use_rerank,
        "answer_model": settings.openrouter_chat_model,
        "judge_model": settings.eval_llm_model,
        "rerank_model": settings.rerank_model if use_rerank else None,
        "n_samples": len(samples),
        "n_failed": len(failed),
        "failed": failed,
        "aggregate": aggregate,
        "per_sample": per_sample,
    }

    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"{stamp}-{label}.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = out_dir / f"{stamp}-{label}.md"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"\nSaved report: {json_path} and {md_path}")
    return report


def render_markdown(report: dict) -> str:
    agg = report["aggregate"]
    lines = [
        f"# Eval report: {report['label']}",
        "",
        f"- Timestamp: {report['timestamp']}",
        f"- Retriever: {report['retriever']} | Rerank: {report['rerank']}",
        f"- Answer model: {report['answer_model']} | Judge: {report['judge_model']}",
        f"- Samples: {report['n_samples']} (failed: {report['n_failed']})",
        "",
        "| Metric | Mean |",
        "|---|---|",
        f"| Output Accuracy (AnswerAccuracy) | {agg['nv_accuracy']:.3f} |",
        f"| Faithfulness | {agg['faithfulness']:.3f} |",
        f"| Context Precision | {agg['llm_context_precision_with_reference']:.3f} |",
        "",
        "## Per-sample scores",
        "",
        "| # | Question | Accuracy | Faithfulness | Ctx Precision |",
        "|---|---|---|---|---|",
    ]
    for i, s in enumerate(report["per_sample"], 1):
        q = s["user_input"].replace("|", "/")[:70]
        lines.append(
            f"| {i} | {q} | {s['nv_accuracy']:.3f} "
            f"| {s['faithfulness']:.3f} | {s['llm_context_precision_with_reference']:.3f} |"
        )
    return "\n".join(lines) + "\n"

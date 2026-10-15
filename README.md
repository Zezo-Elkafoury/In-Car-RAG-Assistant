# Car Manual RAG (MG ZS)

Retrieval-augmented chatbot over MG ZS warning-messages manual pages.
Calls are routed through the **OpenRouter API**; retrieval quality is measured
with **RAGAS**.

## Start here

The main project is the `car_rag` package + CLI — start with `car-rag ask`
below. `notebook.ipynb` is a thin visual demo of the same code, and
`evals/reports/` holds scored RAGAS runs (Output Accuracy, Faithfulness,
Context Precision) comparing dense vs hybrid retrieval.

## Layout

```
pyproject.toml            # uv-managed deps (pinned langchain 0.3 + ragas 0.3 stack)
src/car_rag/
  config.py               # pydantic-settings; everything comes from .env
  prompts.py              # versioned prompt template
  rag.py                  # ingestion, Chroma index, dense/hybrid retrieval, answering
  rerank.py               # OpenRouter /rerank compressor (e.g. cohere/rerank-v3.5)
  eval.py                 # RAGAS harness (AnswerAccuracy, Faithfulness, Context Precision)
  cli.py                  # car-rag ingest | ask | eval
evals/
  datasets/golden.jsonl   # 36 hand-written Q/A pairs from the manual (ground truth)
  generate_dataset.py     # RAGAS TestsetGenerator -> synthetic.jsonl (needs credits)
  run_evals.py            # script entrypoint (same as `car-rag eval`)
  compare_reports.py      # side-by-side table of report JSONs
  reports/                # timestamped JSON + Markdown results
notebook.ipynb            # thin demo importing the package
```

## Setup

```powershell
uv sync
Copy-Item .env.example .env   # then add your OPENROUTER_API_KEY
```

## Usage

```powershell
.venv\Scripts\python -m car_rag.cli ingest --rebuild
.venv\Scripts\python -m car_rag.cli ask "The ABS Fault warning has appeared. What does it mean?"
.venv\Scripts\python -m car_rag.cli ask "..." --retriever hybrid --rerank
.venv\Scripts\python evals/run_evals.py --dataset evals/datasets/golden.jsonl --retriever dense --label dense
.venv\Scripts\python evals/compare_reports.py evals/reports/<dense>.json evals/reports/<hybrid>.json
```

## Evaluation

Three RAGAS metrics, all scored by an LLM judge (`EVAL_LLM_MODEL`):

| Requirement | RAGAS metric |
|---|---|
| Output Accuracy (string evaluation vs reference) | `AnswerAccuracy` |
| Grounded in retrieved context (0-1) | `Faithfulness` |
| Retrieval relevance (0-1) | `LLMContextPrecisionWithReference` |

Dataset = 36 handcrafted golden pairs (`source: golden`) plus 19 RAGAS synthetic
pairs (`evals/datasets/synthetic.jsonl`), merged into `evals/datasets/combined.jsonl`
(55 rows total).

### Results (golden set, mini judge)

| Metric | dense | hybrid | hybrid+rerank |
|---|---|---|---|
| Output Accuracy | 0.944 | 0.946 | 0.958 |
| Faithfulness | 0.987 | 1.000 | 0.994 |
| Context Precision | 0.947 | 0.897 | 0.986 |

`hybrid+rerank` wins on Output Accuracy and Context Precision. Faithfulness is
near-perfect everywhere, as expected on this simple corpus. Full per-sample
reports: `evals/reports/`.

"""Thin script entrypoint for evaluation (same as `car-rag eval`).

Usage:
    python evals/run_evals.py --dataset evals/datasets/golden.jsonl \\
        --retriever hybrid --rerank --label hybrid-rerank
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from car_rag.eval import REPORT_COLUMNS, run_evaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evals/datasets/golden.jsonl")
    parser.add_argument("--retriever", default="dense", choices=["dense", "hybrid"])
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--label", default="run")
    args = parser.parse_args()

    report = run_evaluation(
        dataset_path=args.dataset,
        retriever_mode=args.retriever,
        use_rerank=args.rerank,
        limit=args.limit,
        label=args.label,
    )
    print("\nAggregate scores:")
    for col in REPORT_COLUMNS:
        print(f"  {col}: {report['aggregate'][col]:.3f}")


if __name__ == "__main__":
    main()

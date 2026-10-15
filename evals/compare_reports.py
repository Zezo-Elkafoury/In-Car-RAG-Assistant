"""Compare eval report JSONs, e.g. dense vs hybrid vs hybrid+rerank.

Usage:
    python evals/compare_reports.py evals/reports/<a>.json evals/reports/<b>.json
"""

import json
import sys

LABELS = {
    "nv_accuracy": "Output Accuracy",
    "faithfulness": "Faithfulness",
    "llm_context_precision_with_reference": "Context Precision",
}


def main() -> None:
    reports = []
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as f:
            reports.append(json.load(f))
    if not reports:
        print("Pass report .json paths as arguments.")
        sys.exit(1)

    header = ["Metric"] + [r["label"] for r in reports]
    print(" | ".join(header))
    print(" | ".join(["---"] * len(header)))
    for col, name in LABELS.items():
        print(" | ".join([name] + [f"{r['aggregate'][col]:.3f}" for r in reports]))


if __name__ == "__main__":
    main()

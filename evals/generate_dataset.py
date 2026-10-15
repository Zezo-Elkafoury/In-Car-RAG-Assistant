"""Generate synthetic eval Q/A pairs with RAGAS TestsetGenerator (OpenRouter).

Usage:
    python evals/generate_dataset.py --size 20 --output evals/datasets/synthetic.jsonl
Then merge: cat golden + synthetic into evals/datasets/combined.jsonl
(see evals/merge_datasets.py helper below).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from ragas.testset import TestsetGenerator  # noqa: E402

from car_rag import rag  # noqa: E402
from car_rag.config import get_settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--output", default="evals/datasets/synthetic.jsonl")
    args = parser.parse_args()

    settings = get_settings()
    settings.require_api_key()

    chunks = rag.split_documents(
        rag.load_documents(settings.data_path),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    generator = TestsetGenerator.from_langchain(
        rag.make_chat_llm(settings),  # cheap generator model
        rag.make_embeddings(settings),
    )
    testset = generator.generate_with_langchain_docs(chunks, testset_size=args.size)
    df = testset.to_pandas()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(
                json.dumps(
                    {
                        "user_input": row["user_input"],
                        "reference": row["reference"],
                        "source": "synthetic",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote {len(df)} synthetic rows to {out}")
    print("Review them, then append to evals/datasets/combined.jsonl")


if __name__ == "__main__":
    main()

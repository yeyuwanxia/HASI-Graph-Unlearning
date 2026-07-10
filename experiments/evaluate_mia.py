from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.mia import PrivacyEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MIA AUC from exported before/after logits.")
    parser.add_argument("--logits_before", required=True, help="Path to .npy logits before unlearning.")
    parser.add_argument("--logits_after", required=True, help="Path to .npy logits after unlearning.")
    parser.add_argument("--members", required=True, help="Text file with one forgotten/member node id per line.")
    parser.add_argument("--non_members", required=True, help="Text file with one retained/non-member node id per line.")
    parser.add_argument("--embeddings_before", default=None, help="Optional .npy embeddings before unlearning.")
    parser.add_argument("--embeddings_after", default=None, help="Optional .npy embeddings after unlearning.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    evaluator = PrivacyEvaluator()
    result = evaluator.evaluate_from_logits(
        logits_before=np.load(args.logits_before),
        logits_after=np.load(args.logits_after),
        member_indices=_load_indices(args.members),
        non_member_indices=_load_indices(args.non_members),
        embeddings_before=np.load(args.embeddings_before) if args.embeddings_before else None,
        embeddings_after=np.load(args.embeddings_after) if args.embeddings_after else None,
    ).as_dict()

    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


def _load_indices(path: str) -> list[int]:
    content = Path(path).read_text(encoding="utf-8")
    indices: list[int] = []
    for token in content.replace(",", "\n").splitlines():
        token = token.strip()
        if token:
            indices.append(int(token))
    return indices


if __name__ == "__main__":
    main()

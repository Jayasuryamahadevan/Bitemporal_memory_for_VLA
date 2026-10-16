"""Audit OpenEQA annotations without claiming visual-answer performance.

Download the official data/open-eqa-v0.json and run:
  python -m benchmarks.openeqa_annotations path/to/open-eqa-v0.json

The question-answer file contains no episode frames. This tool reports the
benchmark's question mix only; it cannot score an embodied memory system.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import json


def audit(path: Path) -> dict:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError("OpenEQA annotations must be a list")
    required = {"question", "answer", "category", "question_id", "episode_history"}
    if any(not isinstance(row, dict) or not required <= row.keys() for row in rows):
        raise ValueError("missing required OpenEQA annotation fields")
    ids = [row["question_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question IDs")
    return {
        "benchmark": "OpenEQA v0 annotation audit",
        "questions": len(rows),
        "episode_histories": len({row["episode_history"] for row in rows}),
        "categories": dict(sorted(Counter(row["category"] for row in rows).items())),
        "answer_accuracy_evaluated": False,
        "reason": "episode frames and a grounded answering model are required",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.annotations), indent=2))

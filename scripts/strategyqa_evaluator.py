"""
StrategyQA evaluation script.

This script compares predicted answers against gold labels and reports
the accuracy metric, which is appropriate for the binary yes/no answers
in the StrategyQA task.

Usage:

```
python strategyqa_evaluator.py --pred preds.jsonl --gold dev.jsonl
```

The prediction file should be a JSONL with one object per line.  Each
object must contain an ``answer`` field with a string value of
"yes" or "no".  If an ``id`` or ``question_id`` field is present it will
be used to align predictions with gold entries; otherwise predictions
and gold are matched by line index.
The gold file should be the official dev/test split from StrategyQA,
available via the HuggingFace ``datasets`` library.  Each gold
entry should contain a ``answer"`` field (with "yes" or "no") and,
optionally, an ``id`` or ``question_id`` to match predictions.
"""

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    """Load a JSONL file and return a list of dicts."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Failed to parse JSON on line: {line}\n{err}") from err
            records.append(obj)
    return records


def normalise_answer(ans: str) -> str:
    """Normalise yes/no answers by stripping and lower‑casing."""
    if ans is None:
        return ""
    return ans.strip().lower().replace(".", "")


def evaluate(preds_path: Path, gold_path: Path) -> float:
    """Compute the accuracy between predictions and gold labels."""
    preds = load_jsonl(preds_path)
    gold = load_jsonl(gold_path)

    # Build a lookup for gold answers by id if present
    gold_lookup = {}
    gold_use_idx = True
    for idx, entry in enumerate(gold):
        key = None
        for possible_key in ("id", "question_id", "qid"):
            if possible_key in entry:
                key = entry[possible_key]
                gold_use_idx = False
                break
        if key is None:
            key = idx
        gold_lookup[key] = entry

    correct = 0
    total = 0
    for idx, pred in enumerate(preds):
        # Determine key for this prediction
        key = None
        for possible_key in ("id", "question_id", "qid"):
            if possible_key in pred:
                key = pred[possible_key]
                break
        if key is None:
            key = idx

        gold_entry = gold_lookup.get(key)
        if gold_entry is None:
            # No gold entry found for this key; skip
            continue
        pred_answer = normalise_answer(pred.get("answer") or pred.get("pred"))
        gold_answer = normalise_answer(gold_entry.get("answer") or gold_entry.get("label"))
        if not pred_answer:
            # Missing prediction; count as incorrect
            total += 1
            continue
        if pred_answer == gold_answer:
            correct += 1
        total += 1
    if total == 0:
        raise ValueError("No overlapping predictions/gold labels found.")
    return correct / total


def main():
    parser = argparse.ArgumentParser(description="Compute StrategyQA accuracy.")
    parser.add_argument("--pred", required=True, help="Path to predictions JSONL file")
    parser.add_argument("--gold", required=True, help="Path to gold JSONL file")
    args = parser.parse_args()
    preds_path = Path(args.pred)
    gold_path = Path(args.gold)
    accuracy = evaluate(preds_path, gold_path)
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}% correct)")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Construct baseline and CoT training corpora from teacher outputs.

After running `generate_teacher_responses.py` you will have a JSONL file
containing records with fields `question`, `teacher_thought`, and
`teacher_answer`.  This script reads those records and writes two new JSONL
files:

1. Baseline corpus (Track A): each line contains `prompt` (the original
   question) and `answer` (the teacher’s yes/no).  This will be used to
   fine‑tune a baseline QLoRA model【777585631172426†L42-L45】.
2. CoT corpus (Track B): each line contains `prompt` (question + teacher
   chain‑of‑thought) and `answer`.  This will be used for CoT distillation
   training【777585631172426†L23-L31】.

Example usage:

```
python build_training_corpora.py \
  --input_path data/teacher_outputs.jsonl \
  --baseline_output data/train_baseline.jsonl \
  --cot_output data/train_cot.jsonl
```
"""

import argparse
import json
import os
from typing import Dict, Iterable, Tuple


def load_jsonl(path: str) -> Iterable[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def save_jsonl(path: str, records: Iterable[Dict[str, str]]) -> None:
    # Create parent directories only if a directory component is present.
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_corpora(records: Iterable[Dict[str, str]]) -> Tuple[list, list]:
    baseline = []
    cot = []
    for rec in records:
        question = rec.get("question", "").strip()
        thought = rec.get("teacher_thought", "").strip()
        answer = rec.get("teacher_answer", "").strip()
        if not question or not answer:
            continue
        baseline_prompt = question
        cot_prompt = f"{question} {thought}".strip()
        baseline.append({"prompt": baseline_prompt, "answer": answer})
        cot.append({"prompt": cot_prompt, "answer": answer})
    return baseline, cot


def main() -> None:
    parser = argparse.ArgumentParser(description="Build baseline and CoT training corpora from teacher outputs.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to teacher outputs JSONL.")
    parser.add_argument("--baseline_output", type=str, required=True, help="Where to save baseline corpus JSONL.")
    parser.add_argument("--cot_output", type=str, required=True, help="Where to save CoT corpus JSONL.")
    args = parser.parse_args()
    records = list(load_jsonl(args.input_path))
    baseline, cot = build_corpora(records)
    save_jsonl(args.baseline_output, baseline)
    save_jsonl(args.cot_output, cot)
    print(f"Wrote {len(baseline)} baseline examples to {args.baseline_output}")
    print(f"Wrote {len(cot)} CoT examples to {args.cot_output}")


if __name__ == "__main__":
    main()
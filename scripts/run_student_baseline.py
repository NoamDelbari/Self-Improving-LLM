#!/usr/bin/env python
"""
Generate student draft answers and clarifying questions for StrategyQA examples.

This script loads a base causal‑language model (e.g. a 7 B llama variant) and
produces a short “student draft” for each question in a StrategyQA training
set.  A student draft consists of the model’s initial answer followed by one
or two clarifying sub‑questions.  These drafts will later be passed to GPT‑4
as part of a data‑generation loop described in the project plan【777585631172426†L20-L31】.

The script expects an input JSONL file where each line contains at least
`question` (a yes/no question) and optionally other metadata.  It writes an
output JSONL file with `question` and a new `student_draft` field.

Example usage:

```
python run_student_baseline.py \
  --input_path data/sample_train.jsonl \
  --output_path data/student_drafts.jsonl \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --max_tokens 35
```

Notes:
* The model must be an instruction‑tuned LLM capable of producing
  helpful answers given a short prompt.  For proof‑of‑concept testing, any
  small model (e.g. `gpt2`) can be substituted.
* By default, greedy decoding (`do_sample=False`) is used; you can enable
  sampling and specify a temperature via command‑line flags.
"""

import argparse
import json
import os
from typing import Dict, Iterable

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_dataset(path: str) -> Iterable[Dict[str, str]]:
    """Yield JSON objects from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                yield obj
            except json.JSONDecodeError:
                # Skip malformed lines gracefully
                continue


def save_jsonl(records: Iterable[Dict[str, str]], path: str) -> None:
    """Write a sequence of JSON records to a JSONL file."""
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def generate_student_draft(
    model, tokenizer, question: str, max_tokens: int, do_sample: bool, temperature: float
) -> str:
    """Generate a student draft consisting of an answer and clarifying questions.

    The prompt format loosely follows the project plan: we present the
    question and ask the model to provide a short answer and one or two
    clarifying questions.  We rely on the model’s instruction tuning to
    produce well‑formed output; greedy decoding or low‑temperature sampling
    suffices for this stage.

    Args:
        model: The pretrained causal LM (already moved to the desired device).
        tokenizer: Corresponding tokenizer.
        question: The original yes/no question.
        max_tokens: Maximum number of tokens to generate.
        do_sample: Whether to sample from the distribution (defaults to False).
        temperature: Sampling temperature if `do_sample` is True.

    Returns:
        A string containing the student draft.
    """
    # Build a simple prompt.  We prefix with an instruction to answer and
    # propose clarifying sub‑questions; this can be adjusted to your model.
    prompt = (
        f"Question: {question}\n"
        "Please provide a yes/no answer followed by one or two clarifying questions."\
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    generation_args = {
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "eos_token_id": tokenizer.eos_token_id,
    }
    with torch.no_grad():
        output = model.generate(**inputs, **generation_args)
    # Extract only the newly generated tokens beyond the prompt.
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Remove the prompt portion so only the student draft remains.
    if generated_text.startswith(prompt):
        student_draft = generated_text[len(prompt):].strip()
    else:
        # Fall back to full generation if prefix mismatch.
        student_draft = generated_text.strip()
    # Ensure the draft is at most `max_tokens` tokens long.
    token_ids = tokenizer.encode(student_draft, add_special_tokens=False)
    if len(token_ids) > max_tokens:
        token_ids = token_ids[:max_tokens]
        student_draft = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    return student_draft


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate student drafts for StrategyQA questions.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSONL file containing questions.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSONL file where drafts will be saved.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Name or path of the base model to load (e.g. meta-llama/Llama-2-7b-hf).",
    )
    parser.add_argument("--max_tokens", type=int, default=35, help="Maximum tokens to generate for each draft.")
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Enable sampling (otherwise greedy decoding is used).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (ignored if --do_sample is not set).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Force a specific device (cpu or cuda).  Defaults to cuda if available.",
    )
    args = parser.parse_args()

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer.  Using half precision on CUDA saves memory.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=model_dtype)
    model.to(device)
    model.eval()

    records_out = []
    for item in load_dataset(args.input_path):
        question = item.get("question") or item.get("q") or item.get("Query") or ""
        if not question:
            # Skip entries without a question
            continue
        draft = generate_student_draft(
            model,
            tokenizer,
            question,
            max_tokens=args.max_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
        )
        rec = {
            "question": question,
            "student_draft": draft,
        }
        # Preserve other fields if present (e.g. gold labels)
        for k, v in item.items():
            if k not in rec:
                rec[k] = v
        records_out.append(rec)

    save_jsonl(records_out, args.output_path)
    print(f"Wrote {len(records_out)} student drafts to {args.output_path}")


if __name__ == "__main__":
    main()
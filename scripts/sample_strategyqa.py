"""
Subsample the StrategyQA training dataset.

The project plan recommends using approximately 2000 training examples from
the 2305 available in StrategyQA. This script
randomly selects a specified number of examples (without replacement)
from the input JSONL file and writes them to an output JSONL file.

Usage:

```
python sample_strategyqa.py --input data/raw/strategyqa_train.jsonl \
       --output data/sample_train.jsonl --sample-size 2000 --seed 42
```

If ``--sample-size`` is greater than or equal to the number of examples
in the input file, the script simply copies the entire file to the
output path.
"""

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Sample StrategyQA training data")
    parser.add_argument("--input", required=True, help="Input JSONL file (train split)")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="Number of examples to sample (default: 2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    records = load_jsonl(input_path)
    n_examples = len(records)
    sample_size = args.sample_size
    if sample_size >= n_examples:
        print(
            f"Requested sample size {sample_size} >= number of examples {n_examples}; copying entire dataset."
        )
        sampled = records
    else:
        random.seed(args.seed)
        indices = list(range(n_examples))
        random.shuffle(indices)
        sampled_indices = indices[:sample_size]
        sampled = [records[i] for i in sampled_indices]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(sampled, output_path)
    print(f"Wrote {len(sampled)} examples to {output_path}")


if __name__ == "__main__":
    main()
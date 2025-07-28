"""
Download the StrategyQA dataset from one of several mirrors on Hugging Face and
write each split to JSONL files.

Priority of mirrors (all public & free):
  1. `voidful/StrategyQA`   – full train / test (recommended):contentReference[oaicite:6]{index=6}
  2. `ChilleD/StrategyQA`   – Parquet, no dev split:contentReference[oaicite:7]{index=7}
  3. `wics/strategy-qa`     – test-only mirror:contentReference[oaicite:8]{index=8}

If the chosen mirror has no dedicated *dev* split, we create one by sampling
565 rows (~20 %) from `train`, the split size used in the original paper.
"""

import argparse
import json
import pathlib
import random
from datasets import load_dataset, Dataset

DEFAULT_OUT = pathlib.Path("data/raw")
_MIRRORS = [
    "voidful/StrategyQA",
    "ChilleD/StrategyQA",
    "wics/strategy-qa",
]


def try_load_dataset(path: str) -> Dataset:
    """Attempt to load StrategyQA from the given HF path."""
    try:
        ds = load_dataset(path)
        print(f"✓ loaded '{path}' with splits: {list(ds.keys())}")
        return ds
    except Exception as exc:  # pragma: no cover
        print(f"✗ failed '{path}': {exc}")
        raise


def main(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    for mirror in _MIRRORS:
        try:
            ds_dict = try_load_dataset(mirror)
            break
        except Exception:
            continue
    else:  # pragma: no cover
        raise RuntimeError("Unable to load StrategyQA from any known mirror.")

    # If the mirror lacks a validation/dev split, sample one from train.
    if "validation" not in ds_dict and "dev" not in ds_dict:
        dev_size = 565  # matches original dataset spec
        split = ds_dict["train"].train_test_split(
            test_size=dev_size, seed=42, shuffle=True
        )
        ds_dict["train"], ds_dict["validation"] = split["train"], split["test"]
        print(f"Created synthetic dev split with {len(ds_dict['validation'])} rows.")

    # Save each available split as JSONL
    for split, dataset in ds_dict.items():
        out_path = out_dir / f"strategyqa_{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for ex in dataset:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote {len(dataset):>5} rows -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUT),
        help="Directory where raw JSONL splits will be saved",
    )
    main(parser.parse_args())

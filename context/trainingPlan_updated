# Self-Improving LLM Training Plan (Updated)

_Last Updated: Phase B Part 2 integrated (noAsk **69%**, ask **66%**); Phase C unblocked_

## Project Overview

This project implements a self-improving Large Language Model system on StrategyQA. The goal is to fine-tune a ~7B student model to learn asking clarifying sub-questions, harvest GPT‑4 “teacher” answers, and distill those chains of thought back into itself (System‑2 → System‑1).

### Architecture
- **Base Model**: Phi-3.5-mini-instruct (≈7B)
- **Training Method**: QLoRA (4‑bit) with PEFT
- **Hardware**: Single RTX‑4060 (8 GB)
- **Dataset**: StrategyQA (binary yes/no commonsense questions)

## Training Pipeline

1. **Phase A** — Baseline SFT (Track A): `Q → answer`
2. **Phase B** — CoT Distillation (Track B): `Q + teacher CoT → answer`
   - **Part 1** (initial run): failed (regressed)
   - **Part 2** (new Colab): success with two CoT variants (see below)
3. **Phase C** — Preference Alignment (DPO): reward self‑questioning behavior

---

## Data Generation & Preparation (Completed)

- **Sampling**: 200 train examples (configurable)
- **Student Drafts**: Generated with Phi‑3.5‑mini using structured prompts
- **Teacher Responses**: GPT‑4 “teaching” format (short steps + final yes/no)
- **Validation**: 137/200 valid (68.5%)

---

## Phase A — Baseline (✅ Completed)

**Config**
```python
PHASE_A_CONFIG = {
    'model_name': 'microsoft/Phi-3.5-mini-instruct',
    'train_file': 'data/train_baseline.jsonl',
    'output_dir': 'models/baseline_phaseA',
    'max_length': 2048,
    'num_epochs': 3,
    'batch_size': 2,
    'gradient_accumulation_steps': 16,
    'learning_rate': 2e-4,
    'use_4bit': True,
}
```

**Results**
- **Dev Accuracy**: **59.4%** (target ~60% met)
- **LoRA**: r=16, α=32, dropout=0.1 on attention+MLP proj modules
- **GPU**: ~2.7 GB with 4‑bit

---

## Phase B — CoT Distillation

### Part 1 (❌ initial attempt — regression)
**Setup**
- Base: `models/baseline_phaseA/`
- Train file: `data/train_cot.jsonl` (137 examples)
- Format: `Question + Student Draft + Teacher Reasoning → Answer`

**Observed Issues**
- **Accuracy**: **51.5%** (−7.9pp vs baseline)
- **Skew**: 71.5% “No” after filtering → bias to “No”
- **Data quality**: ~31.5% teacher↔gold disagreement (filtered out)
- **Format mismatch**: train/eval templates differed
- **Net effect**: no CoT benefit; incomplete/truncated generations

### Part 2 (✅ new Colab experiment — recovered)
**Goal.** Isolate effect of conditioning teacher CoT on the student’s questions.

**Variants**
1. **noAsk teacher‑CoT** — Teacher reasoning independent of student questions.  
   Train schema: `Question + Teacher CoT → Answer`
2. **ask teacher‑CoT** — Teacher reasoning written in response to the student’s clarifying questions.  
   Train schema: `Question + Teacher CoT (re: student questions) → Answer`

**Results (dev accuracy)**
| Variant | Schema | Accuracy |
|---|---|---:|
| **noAsk teacher‑CoT** | `Q + Teacher CoT → A` | **69%** |
| **ask teacher‑CoT** | `Q + Teacher CoT (conditioned on student Qs) → A` | **66%** |

**Takeaways**
- Both variants beat Phase A (59.4%).  
- **noAsk (69%) > ask (66%)** in our current setup. Likely reasons:
  - Student‑conditioned CoT inherits noise from weak student drafts.
  - Longer inputs (ask variant) increase truncation/format drift risk.
  - Extra structure may need preference alignment (DPO) to shine.

**Decision**
- Use **noAsk@69%** checkpoint as **Phase B best** and **start point for Phase C (DPO)**.
- Retain **ask@66%** for ablations and post‑DPO re‑tests.

**Actions**
- Standardize templates so train/eval exactly match.
- Keep CoT short (1–3 steps + final) to fit ≤128–256 tokens.

**Config (Part 2)**
```python
PHASE_B_CONFIG = {
    'base_checkpoint': 'models/baseline_phaseA/',
    'train_file': 'data/train_cot_noask.jsonl',  # swap to *_ask.jsonl for ask variant
    'output_dir': 'models/cot_phaseB_noask_69/',
    'max_length': 1024,
    'num_epochs': 3,
    'batch_size': 2,
    'gradient_accumulation_steps': 16,
    'learning_rate': 2e-4,
    'use_4bit': True,
    'bf16': True,
}
```

---

## Phase C — Preference-Based Alignment (DPO)

**Objective.** Align the student to **prefer** responses that show brief self‑questioning (1–2 clarifying mini‑questions) before committing to Yes/No.

**Inputs**
- **Base**: `models/cot_phaseB_noask_69/` (best CoT)
- **Pairs** (`data/pairs.jsonl`):
  ```json
  {"prompt": "<Q>", "chosen": "<teacher_resp with 1–2 mini‑questions + final>", "rejected": "<student_draft>"}
  ```

**Config**
```python
PHASE_C_CONFIG = {
    'base_checkpoint': 'models/cot_phaseB_noask_69/',
    'pairs_file': 'data/pairs.jsonl',
    'output_dir': 'models/dpo_phaseC/',
    'beta': 0.1,
    'num_epochs': 1,          # short pass
    'batch_size': 2,
    'learning_rate': 1e-5,
}
```

**Procedure**
1. Build preference pairs (trim to concise clarifiers + final answer).
2. Train with TRL `DPOTrainer` (β≈0.1).
3. Re‑evaluate on StrategyQA dev; manual check for “ask‑then‑answer” behavior.

**Expected**
- Small accuracy bump vs 69% and visibly more self‑questioning outputs.

---

## Validation & Ablations

| Test | Procedure | Success Criterion |
|---|---|---|
| **No‑draft ablation** | Prompt = `Q` only | ≥3pp drop vs full prompt |
| **Shuffled‑draft** | Pair `Q` with random draft | Additional drop |
| **ask vs noAsk (post‑DPO)** | Evaluate both checkpoints after DPO | ask closes the gap or wins |
| **Attention attribution** | IG/attn mass on clarifier tokens | ≥20% |
| **GPT‑judge (optional)** | Judge usefulness of clarifiers | +10% win‑rate |

---

## Resources & Timeline

**Compute**
- Phase A ≈ 2 GPU‑hours • Phase B ≈ 2 • Phase C ≈ 1–2 → **~5–6 h total** (8 GB ok).

**Budget**
- GPT‑4: ~30k–300k tokens depending on scale.

**7‑Day Plan**
| Day | Task |
|---|---|
| 1 | Repo setup; dataset sample |
| 2 | Prompting & scripts |
| 3 | Run student+teacher loop; QA outputs |
| 4 | Build Track A/B; Phase A train |
| 5 | Phase A eval; Phase B Part 2 train (noAsk/ask) |
| 6 | Evaluate (69%/66%); prep DPO pairs |
| 7 | DPO train from noAsk@69; final eval & report |

---

## Current Status (Executive)

- ✅ **Phase A**: 59.4% (baseline met)
- ✅ **Phase B Part 2**: **69% (noAsk)**, **66% (ask)** — recovered & improved
- ▶️ **Phase C**: Ready (start from `cot_phaseB_noask_69`)

**Recommendation**: Proceed to **DPO** from the **69%** checkpoint; keep **ask** variant for post‑DPO comparison.

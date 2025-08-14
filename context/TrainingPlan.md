# Self-Improving LLM Training Plan

## Project Overview

This project implements a self-improving Large Language Model system based on StrategyQA dataset. The goal is to fine-tune a 7B parameter student model to learn asking clarifying sub-questions, harvest GPT-4 "teacher" answers, and distill those chains of thought back into itself (System-2 → System-1).

### Architecture
- **Base Model**: Microsoft Phi-2/Phi-3.5-mini-instruct (7B parameters)
- **Training Method**: QLoRA (4-bit quantization) 
- **Hardware**: Single RTX-4060 (8GB GPU)
- **Dataset**: StrategyQA (binary yes/no commonsense questions)

## Training Pipeline Overview

The training consists of three distinct phases:

1. **Phase A**: Baseline Supervised Fine-Tuning (Track A) - Direct question-answer pairs
2. **Phase B**: Chain-of-Thought (CoT) Distillation (Track B) - Incorporating teacher reasoning
3. **Phase C**: Preference-based Alignment (DPO) - Rewarding self-questioning behavior

## Current Implementation Status

### ✅ Data Generation & Preparation (Completed)
- **Dataset**: StrategyQA with 2,305 train / 565 dev / 490 test examples
- **Sampling**: 200 training examples (configurable via TRAIN_SAMPLES)
- **Student Draft Generation**: Using Phi-3.5-mini-instruct with structured prompts
- **Teacher Response Generation**: Using GPT-4 with educational reasoning format
- **Data Validation**: 68.5% data quality rate (137/200 valid examples)

### ✅ Phase A: Baseline Training (Completed)
**Status**: ✅ Training and evaluation completed successfully

**Configuration**:
```python
PHASE_A_CONFIG = {
    'model_name': 'microsoft/Phi-3.5-mini-instruct',  # Updated model
    'train_file': 'data/train_baseline.jsonl',
    'output_dir': 'models/baseline_phaseA',
    'max_length': 2048,
    'num_epochs': 3,
    'batch_size': 2,  # Adjusted for GPU memory
    'gradient_accumulation_steps': 16,
    'learning_rate': 2e-4,
    'use_4bit': True
}
```

**Training Details**:
- **Dataset**: 137 validated question-answer pairs
- **Format**: `Question: {question}\nAnswer: {yes/no}`
- **LoRA Config**: r=16, alpha=32, dropout=0.1
- **Target Modules**: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj
- **Memory Usage**: ~2.7GB GPU memory
- **Trainable Parameters**: 8.9M (0.23% of total 3.8B parameters)

**Actual Results**: 
- **Test Accuracy**: 59.4% (408/687 examples) ✅ **TARGET MET**
- **Model Performance**: Successfully meets ~60% baseline target
- **Example Predictions**: Model generates coherent Yes/No answers
- **Results File**: `results_phase_a.json` saved for comparison

## Phase B: CoT Distillation (❌ FAILED - Critical Issues Identified)

**Status**: ❌ Training completed but FAILED to achieve target performance

### Input Artifacts
- **Base Model**: `models/baseline_phaseA/` (Phase A checkpoint)
- **Dataset**: `data/train_cot.jsonl` (137 validated examples)
- **Format**: Question + Student Draft + Teacher Reasoning → Answer

### Training Configuration (Used)
```python
PHASE_B_CONFIG = {
    'base_checkpoint': 'models/baseline_phaseA/',
    'train_file': 'data/train_cot.jsonl',
    'output_dir': 'models/cot_phaseB/',
    'max_length': 2048,
    'num_epochs': 3,
    'batch_size': 2,  # Adjusted for GPU memory
    'gradient_accumulation_steps': 16,
    'learning_rate': 2e-4,
    'use_4bit': True,
    'fp16': False,  # Disabled for PEFT compatibility
    'bf16': True   # Used when supported
}
```

### Actual Procedure (Completed)
1. ✅ Loaded Phase A checkpoint successfully
2. ✅ Applied PEFT model loading from Phase A
3. ✅ Trained on CoT format with comprehensive error handling
4. ✅ Saved model to `models/cot_phaseB/`
5. ❌ **PERFORMANCE REGRESSION OCCURRED**

### **CRITICAL FAILURE ANALYSIS**

**Actual Results**: 
- **CoT Model Accuracy**: 51.5% (354/687 examples)
- **Phase A Baseline**: 59.4% (408/687 examples)  
- **Performance Regression**: **-7.9 percentage points** ❌
- **Target**: 67-70% (+7-10pp) - **MISSED BY 15.5-18.5pp**

**Root Cause Analysis**:

#### 1. **Severe Class Imbalance Problem**
```
Original Dataset: 91 Yes (45.5%), 109 No (54.5%) - Relatively Balanced
Training Data:    39 Yes (28.5%), 98 No (71.5%) - SEVERELY SKEWED
```
- **2.5:1 bias toward "No" answers** caused model to default to "No"
- Lost 63 examples (31.5%) due to GPT-4/ground-truth disagreements
- Aggressive filtering created artificial class imbalance

#### 2. **Data Quality Issues**
- **Data Loss**: 63/200 examples (31.5%) filtered out for teacher errors
- **Quality Rate**: Only 68.5% of teacher responses matched ground truth
- **Selection Bias**: Filtering may have removed harder "Yes" cases

#### 3. **Format Mismatch Problem**
- **Training Format**: Full teacher reasoning with structured format
- **Evaluation Format**: Simplified reasoning context  
- **Mismatch Impact**: Model trained on different input than evaluation

#### 4. **No CoT Benefit Detected**
- **With Reasoning**: 51.5% accuracy
- **Without Reasoning**: 51.5% accuracy  
- **CoT Benefit**: 0.0pp - **ZERO IMPROVEMENT**
- **Interpretation**: Model not effectively using reasoning context

### **Evidence from Evaluation**
```
Sample Generations:
Q: "Was ship that recovered Apollo 13 named after a Wo..."
Generated: "No, the ship that"  
Expected: "Yes" ❌

Q: "Is the tibia necessary to win the Stanley Cup?"
Generated: "No, the tibia"
Expected: "Yes" ❌

Q: "Could the Powerpuff Girls make the background to th..."  
Generated: "To determine whether the"
Expected: "Yes" ❌
```

**Pattern**: Model defaults to "No" and shows truncated/incomplete generations

### **REQUIRED FIXES FOR PHASE B**

**Before proceeding to Phase C, these critical issues must be resolved:**

#### **Fix 1: Address Class Imbalance**
```python
# Option A: Use all teacher data (remove filtering)
# Option B: Balance classes by upsampling minority class
# Option C: Implement weighted loss function
```

#### **Fix 2: Improve Data Quality**
```python
# Better teacher prompts for higher agreement rate
# Multiple teacher attempts with consensus
# Less aggressive filtering criteria
```

#### **Fix 3: Fix Format Consistency**  
```python
# Match training format with evaluation format
# Ensure reasoning context is properly utilized
# Test different reasoning prompt structures
```

#### **Fix 4: Enhanced Training**
```python
# Lower learning rate for stability
# More epochs with early stopping
# Better validation methodology
```

**Decision Point**: 
- **Option 1**: Fix Phase B before proceeding ✅ **RECOMMENDED**
- **Option 2**: Skip directly to Phase C using Phase A baseline
- **Option 3**: Redesign the entire CoT approach

---

## Phase C: DPO Alignment (⏳ BLOCKED - Pending Phase B Fix)

### Input Artifacts
- **Base Model**: `models/cot_phaseB/` (Phase B checkpoint)
- **Preference Pairs**: `data/pairs.jsonl` with fields:
  - `prompt`: question + "Student draft: " + student_draft
  - `chosen`: teacher_final (correct answer)
  - `rejected`: student_draft (when incorrect) or flipped preference

### Training Configuration
```python
PHASE_C_CONFIG = {
    'base_checkpoint': 'models/cot_phaseB/',
    'pairs_file': 'data/pairs.jsonl',
    'output_dir': 'models/dpo_phaseC/',
    'beta': 0.1,
    'num_epochs': 4,
    'batch_size': 2,
    'learning_rate': 1e-5,
}
```

### Procedure
1. Wrap CoT model in TRL DPOTrainer with β=0.1
2. **Cal-DPO**: Subtract batch-mean log-gap μ each step
3. **Dr-DPO**: Drop top 15% noisy pairs by disagreement score
4. **Training Schedule**:
   - Epoch 1: β=0.05, no Cal-DPO (warm-up)
   - Epochs 2-3: β=0.10, Cal-DPO on (core)
   - Epoch 4: β=0.10 + Dr-DPO filter (robust)
5. Optional Self-Guided loop: regenerate drafts, rebuild pairs, repeat

### Success Metrics
- **Target**: ≥+10pp over Phase A and ≥+3pp vs No-draft ablation
- **Validation Metrics**:
  - Attention attribution ≥20% on draft tokens
  - GPT-4 question-quality win-rate +10%

## Validation Experiments

To prove the model learns by asking clarifying questions:

| Test | Procedure | Success Criterion |
|------|-----------|-------------------|
| **No-draft ablation** | prompt = Q only | ≥3pp accuracy drop vs full-draft |
| **Shuffled-draft** | pair Q with random draft | Further accuracy drop |
| **Attention attribution** | Integrated-gradients mass on draft tokens | ≥20% after DPO |
| **GPT-4 question-quality** | Judge usefulness of clarifiers | +10% win-rate |

## Data Generation Details

### Student Draft Format
```
Answer: <Yes/No> - <brief reasoning>
Questions: <focused question>? <key uncertainty>?
```

### Teacher Response Format
```
## Teaching Analysis
[2-3 sentences acknowledging student's approach]

## Step-by-Step Reasoning
[Numbered steps for educational reasoning]

## Final Assessment
Based on this analysis, the answer is: **[YES/NO]**
```

### Training Corpora
- **Track A (Baseline)**: `(question → answer)` - 137 examples
- **Track B (CoT)**: `(question + teacher reasoning → answer)` - 137 examples

## Resource Requirements & Timeline

### Compute Requirements
- **Phase A**: ~2 GPU-hours on RTX-4060
- **Phase B**: ~2 GPU-hours on RTX-4060  
- **Phase C**: ~3 GPU-hours on RTX-4060
- **Total**: ~7 hours (fits 7-day schedule)

### Budget
- **GPT-4 Calls**: 200 × 150 tokens ≈ 30k tokens → ~$1.20
- **GPU**: Single RTX-4060 (8GB) sufficient for all phases

### Timeline (7 Days)
| Day | Task |
|-----|------|
| 1 | ✅ Repo setup, dataset fetch & sampling |
| 2 | ✅ Prompt engineering, generation script |
| 3 | ✅ Run GPT-4 loop, sanity-check outputs |
| 4 | ✅ Filter & format training sets, Phase A training |
| 5 | ✅ Phase A evaluation (59.4% accuracy - target met), ready for Phase B |
| 6 | 🔄 Phase B training (CoT distillation), Phase B evaluation |
| 7 | Phase C (DPO) training, final evaluation, write report & slides |

## Current Results

### Phase A Results (✅ Completed)
- **Training**: Completed successfully
- **Model Size**: 8.9M trainable parameters (0.23% of total 3.8B)
- **Memory Usage**: 2.7GB GPU memory with 4-bit quantization
- **Test Accuracy**: 59.4% (408/687 examples) - **TARGET ACHIEVED**
- **Performance**: Successfully meets baseline target (~60%)
- **Model**: Phi-3.5-mini-instruct with LoRA fine-tuning

### Data Quality Metrics
- **Total Processed**: 200 teacher responses
- **Valid Records**: 137 (68.5% success rate)
- **Invalid Answers**: 0 (all Yes/No format)
- **Incorrect vs Ground Truth**: 63 (31.5% teacher errors)

## **CURRENT PROJECT STATUS - CRITICAL REVIEW NEEDED**

### **Completed Phases**
1. ✅ **Data Generation**: 200 samples → 137 validated examples (68.5% quality)
2. ✅ **Phase A (Baseline)**: 59.4% accuracy - **TARGET MET** ✅
3. ❌ **Phase B (CoT)**: 51.5% accuracy - **FAILED** (-7.9pp regression)

### **Critical Issues Identified**
| Issue | Impact | Status |
|-------|--------|--------|
| **Class Imbalance** | 71.5% "No" answers → model bias | ❌ **BLOCKING** |
| **Data Quality** | 31.5% teacher-truth disagreement | ❌ **BLOCKING** |
| **Format Mismatch** | Training ≠ evaluation format | ❌ **BLOCKING** |
| **No CoT Benefit** | 0.0pp improvement with reasoning | ❌ **BLOCKING** |

### **Next Steps - URGENT DECISION REQUIRED**

**Option 1: Fix Phase B (RECOMMENDED)** 🎯
- **Priority**: HIGH - Fix class imbalance and data quality
- **Timeline**: 1-2 days to reimplement Phase B properly
- **Success Criteria**: Achieve 67-70% accuracy target

**Option 2: Skip to Phase C with Phase A Model**
- Use Phase A (59.4%) as base for DPO alignment
- Accept that CoT distillation approach failed
- Focus on DPO for final performance gain

**Option 3: Complete Project Redesign**
- Rethink the CoT approach entirely
- Consider different reasoning formats
- May require significant timeline extension

### **Resource Status**
- ✅ **Infrastructure**: Working (4-bit LoRA, PEFT pipeline)
- ✅ **Phase A Model**: Solid 59.4% baseline preserved
- ❌ **Phase B Model**: Degraded performance, needs fixing
- ✅ **Data Pipeline**: Functional but needs quality improvements

### **Project Timeline Impact**
```
Original Plan: 7 days
Current Status: Day 5-6
Remaining: 1-2 days

Critical Path: Fix Phase B → Phase C → Final evaluation
Risk: May not complete Phase C if Phase B takes too long
```

## Configuration Files

### Environment Variables (.env)
```bash
# Dataset parameters
DATASET_NAME=voidful/StrategyQA
TRAIN_SAMPLES=200
RANDOM_SEED=42

# Model parameters  
MODEL_NAME=microsoft/phi-2
MAX_NEW_TOKENS=35
BATCH_SIZE=8
USE_4BIT=True
MAX_SEQ_LENGTH=2048

# GPT-4 parameters
OPENAI_API_KEY=your_api_key_here
GPT4_MODEL=gpt-4.1-nano-2025-04-14
GPT4_MAX_TOKENS=150
GPT4_TEMPERATURE=0.3
DRY_RUN=False
```

### File Structure
```
Self-Improving-LLM/
├── data/
│   ├── raw/                    # Original dataset files
│   ├── sample_train.jsonl      # Sampled training data
│   ├── student_drafts.jsonl    # Generated student responses
│   ├── teacher_outputs.jsonl   # GPT-4 teacher responses  
│   ├── train_baseline.jsonl    # Track A training data
│   └── train_cot.jsonl         # Track B training data
├── models/
│   ├── baseline_phaseA/        # Phase A checkpoint
│   ├── cot_phaseB/            # Phase B checkpoint (planned)
│   └── dpo_phaseC/            # Phase C checkpoint (planned)
├── notebooks/
│   └── project_notebook.ipynb # Main implementation
└── TrainingPlan.md            # This document
```

---

## **EXECUTIVE SUMMARY**

### **Project Goals vs Actual Results**
| Phase | Target | Actual | Status |
|-------|--------|---------|---------|
| **Phase A (Baseline)** | ~60% | 59.4% | ✅ **SUCCESS** |
| **Phase B (CoT)** | 67-70% (+7-10pp) | 51.5% (-7.9pp) | ❌ **CRITICAL FAILURE** |
| **Phase C (DPO)** | +10pp additional | Not started | ⏳ **BLOCKED** |

### **Key Findings**
1. **Baseline Training Works**: Phi-3.5-mini with LoRA successfully learns StrategyQA
2. **CoT Distillation Failed**: Multiple critical issues caused performance regression  
3. **Data Quality Critical**: 31.5% teacher-ground truth disagreement is too high
4. **Class Balance Matters**: 71.5% "No" bias severely impacted model behavior

### **Technical Achievements**
- ✅ **4-bit LoRA Pipeline**: Efficient training on 8GB GPU (2.7GB memory)
- ✅ **Data Generation**: Automated student-teacher pipeline with GPT-4
- ✅ **Structured Evaluation**: Comprehensive accuracy metrics and analysis
- ✅ **Error Analysis**: Detailed root cause identification for failures

### **Lessons Learned**
1. **Aggressive Filtering Harmful**: Removing 31.5% of data created class imbalance
2. **Format Consistency Critical**: Training/evaluation mismatch prevents learning
3. **Teacher Quality Variable**: GPT-4 disagrees with ground truth frequently  
4. **CoT Benefit Not Automatic**: Simply adding reasoning doesn't guarantee improvement

### **Recommendation**
**Fix Phase B before proceeding** - The identified issues are addressable with proper data handling and training methodology. Phase A provides a solid foundation (59.4%) that makes the project technically viable.

---

*Last Updated: Phase B evaluation completed - Critical issues identified*
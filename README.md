Self‑Improving LLM Project
This repository contains the code for a self‑improving large language model (LLM) project. The goal of this project is to fine‑tune a 7 B parameter student model so that it learns to ask clarifying sub‑questions, harvest step‑by‑step answers from a stronger teacher model (GPT‑4), and then distil those chains of thought back into itself. In effect, we want to transform a System‑2 reasoning process (the teacher’s chain of thought) into a more efficient System‑1 model that produces accurate answers with minimal latency.
Two datasets are offered by the project plan:
•	Option 1 — StrategyQA (recommended): binary yes/no commonsense questions that rely on a few supporting facts. This dataset has short contexts, which keeps token costs low, and the single‑token gold label makes evaluation trivial (accuracy).
•	Option 2 — HotpotQA (alternative): open‑domain, paragraph‑level multi‑hop questions with supporting‑fact supervision. This variant is more resource‑intensive — it requires retrieval, longer sequences, and more memory — and uses EM/F1 as the evaluation metric.
By default we start with StrategyQA because it is cheaper to run and easier to evaluate. If you decide to try the HotpotQA version later, the same pipeline applies with adjustments noted in the project plan.
Repository structure
The code is organised into the following top‑level directories (some may be created later as needed):
Directory	Purpose
scripts/	Data download, preprocessing, prompt generation, training and evaluation scripts.
data/	Raw and processed dataset files.
models/	Saved checkpoints for baseline, CoT‑distilled and optional DPO models.
configs/	YAML/JSON configuration files for training runs.
notebooks/	Optional Colab notebooks for running end‑to‑end pipelines.
Dependencies
Dependencies are listed in requirements.txt. Install them via:
pip install -r requirements.txt
Major libraries include:
•	Transformers (from Hugging Face) for model loading and QLoRA fine‑tuning.
•	Datasets for downloading and handling the StrategyQA/HotpotQA datasets.
•	PEFT for parameter‑efficient fine‑tuning (LoRA/QLoRA).
•	Accelerate to manage distributed and mixed‑precision training.
•	TRL for optional Direct Preference Optimisation (DPO).
•	OpenAI for accessing GPT‑4 when generating teacher chains of thought.
Running the pipeline
1.	Download and sample the dataset — scripts in scripts/ can fetch StrategyQA, subsample the training split, and save the results into data/.
2.	Generate student drafts and teacher responses — use the provided prompt template to create clarifying questions and call GPT‑4. The output is split into two parallel corpora: Track A (question → answer) and Track B (question + teacher chain of thought → answer).
3.	Fine‑tune the student model — train a baseline model on Track A and a CoT‑distilled model on Track B using QLoRA for three epochs with a learning rate of 2e‑4. Optionally perform a DPO fine‑tuning step using preference pairs where the teacher answer is preferred over the student draft.
4.	Evaluate — evaluate on the dev/test splits using the provided strategyqa_evaluator.py script and aim for a +10–15 percentage‑point improvement in accuracy over the baseline.
5.	Report — summarise methodology, results and lessons learned in a report and slide deck at the end of the project.
This README will evolve as we implement each step. For details on the high‑level plan and timeline, see the PDF included in this repository.
________________________________________

#!/usr/bin/env python
"""
Usage:
    python scripts/build_tracks.py \
        --raw student_drafts.jsonl \
        --out data/strategyqa \
        --model gpt-4o-mini \
        --temp 0
Writes:
    train_A.jsonl   (question → gold answer)
    train_B.jsonl   (question + CoT → gold answer)
"""
import json, argparse, pathlib, time
import dotenv, os, openai, tqdm
import re

# ---- Gold-label lookup ---------------------------------------------------
def norm(text: str) -> str:
    """Collapse whitespace so questions match across files."""
    return " ".join(text.split())

LABEL_LOOKUP = {}
with open("data/sample_train.jsonl", encoding="utf-8") as f:
    for row in map(json.loads, f):
        ans_raw = str(row["answer"]).strip().lower()
        if ans_raw in {"yes", "true", "1"}:
            label = 1
        elif ans_raw in {"no", "false", "0"}:
            label = 0
        else:
            # skip rows with weird labels
            continue
        LABEL_LOOKUP[norm(row["question"])] = label
# -------------------------------------------------------------------------


dotenv.load_dotenv()

client = openai.OpenAI()        # uses OPENAI_API_KEY from env

import re, time

# ------------------------------------------------------------------------
# Constants (tweak as you like)
GPT4_MODEL        = "gpt-4o-mini"   # or "gpt-4o" for higher accuracy
GPT4_MAX_TOKENS   = 400             # keeps cost predictable
GPT4_TEMPERATURE  = 0               # deterministic answers
# ------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert teacher helping a student AI model learn to reason through
complex yes/no questions. Your role is to:

1. Analyze the student's draft answer and clarifying questions
2. Provide clear, step-by-step reasoning (2-4 concise steps maximum)
3. Address the student's specific concerns when they raise valid points
4. Keep your reasoning focused and under 200 words
5. Always conclude with a confident final Yes/No answer

Your reasoning should be educational but concise - the student model needs to learn from clear,
digestible explanations."""

# ------------------------------------------------------------------------
def call_teacher(model, question, draft, temp, retries=0):
    """
    Ask GPT-4(o) and return (cot_text, yes_or_no_int).
    If parsing fails after all retries → returns (None, None).
    """

    user_prompt = f"""Question: {question}

                    Student's draft attempt: {draft}

                    Please provide step-by-step reasoning and your final Yes/No answer."""

    try:
        rsp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=GPT4_MAX_TOKENS,
            temperature=GPT4_TEMPERATURE,
        )
    except openai.OpenAIError as e:
        # retry on transient errors
        if retries:
            time.sleep(1); return call_teacher(model, question, draft, temp, retries - 1)
        print("[WARN] API failure:", e); return None, None

    txt = rsp.choices[0].message.content

    # ---------- split & parse ------------------------------------------
    cot, tail = txt.rsplit("Final", 1) if "Final" in txt else (txt, "")
    tail = tail.lower()

    if "yes" in tail:
        return cot.strip(), 1
    if "no" in tail:
        return cot.strip(), 0

    # Second-chance regex anywhere in last line
    last = txt.strip().splitlines()[-1].lower()
    m = re.search(r'\b(yes|no)\b', last)
    if m:
        return cot.strip(), 1 if m.group(1) == "yes" else 0

    # Could not parse → retry / skip
    if retries:
        time.sleep(1); return call_teacher(model, question, draft, temp, retries - 1)

    print("[WARN] unparseable reply, skipping.")
    return None, None
# ------------------------------------------------------------------------


# ── NEW main() ────────────────────────────────────────────────────────────
def main(raw, out, model, temp, pause, no_clarif, tag):
    """
    Create Track-A and Track-B corpora.

    Track-B variant is controlled by:
        --no_clarif   → teacher sees only the question   (NoAsk)
        default       → teacher also sees student draft (SelfAsk)

    Files are written under  <out>/<tag>/  so multiple variants coexist.
    """
    out_dir = pathlib.Path(out) / tag          # e.g. data/strategyqa/selfask
    out_dir.mkdir(parents=True, exist_ok=True)

    A = open(out_dir / "train_A.jsonl", "w", encoding="utf-8")
    B = open(out_dir / "train_B.jsonl", "w", encoding="utf-8")

    bad = total = 0
    for line in tqdm.tqdm(open(raw, encoding="utf-8")):
        row = json.loads(line)
        total += 1

                # choose draft text (empty if --no_clarif)
        draft_text = "" if no_clarif else row["student_draft"]

        cot, teacher_ans = call_teacher(
            model=model,
            question=row["question"],
            draft=draft_text,     # <-- fixed
            temp=temp,
        )


        gold = LABEL_LOOKUP.get(norm(row["question"]))
        if gold is None: 
            print("[WARN] question not found in label table"); continue

        if teacher_ans != gold:
            print("\nQUESTION:", row['question'][:120])
            print("TEACHER  :", teacher_ans, "| GOLD :", gold)
            bad += 1
            continue        # <- uncomment to drop mismatched rows

        # Track A (always written; tiny, no harm)
        json.dump({"question": row["question"], "answer": gold}, A); A.write("\n")

        # Track B
        full = row["question"] + "\n\nThought:\n" + cot
        json.dump({"question": full, "answer": gold}, B); B.write("\n")
        time.sleep(pause)

    A.close(); B.close()
    print(f"Mismatches: {bad}/{total} = {bad/total:.1%}")
# ──────────────────────────────────────────────────────────────────────────


# ── NEW argparse block  (replace the old one) ─────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw",   default="data/student_drafts.jsonl",
                    help="JSONL with question, student_draft, gold answer")
    ap.add_argument("--out",   default="data/strategyqa",
                    help="parent output directory")
    ap.add_argument("--tag",   default="selfask",
                    help="sub-folder name inside --out (e.g. selfask, noask)")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temp",  type=float, default=0)
    ap.add_argument("--pause", type=float, default=0.5,
                    help="seconds to sleep between API calls")
    ap.add_argument("--no_clarif", action="store_true",
                    help="omit student clarifying questions from teacher prompt")
    main(**vars(ap.parse_args()))
# ──────────────────────────────────────────────────────────────────────────

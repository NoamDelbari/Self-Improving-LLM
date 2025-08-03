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

# ---- at the top, after imports ----
LABEL_LOOKUP = {}
with open("data/sample_train.jsonl", encoding="utf-8") as f:
    for row in map(json.loads, f):
        LABEL_LOOKUP[row["question"]] = 1 if row["answer"] else 0
# -----------------------------------

dotenv.load_dotenv()

client = openai.OpenAI()        # uses OPENAI_API_KEY from env

PROMPT = """You are a precise reasoning engine.
Question: {q}

Student clarifying questions:
{cl}

Answer step by step; finish with 'Final: Yes' or 'Final: No'."""

# --------------------------------------------------------------------
def call_teacher(model, question, clarif, temp):
    prompt = PROMPT.format(q=question, cl=clarif)
    rsp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
    )
    txt = rsp.choices[0].message.content
    cot, final = txt.rsplit("Final:", 1)
    ans = 1 if "yes" in final.lower() else 0
    return cot.strip(), ans

def main(raw, out, model, temp, pause):
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    A = open(out/"train_A.jsonl", "w"); B = open(out/"train_B.jsonl", "w")
    bad = total = 0
    for line in tqdm.tqdm(open(raw, encoding="utf-8")):
        row = json.loads(line)
        total += 1

        cot, teacher_ans = call_teacher(model,
                                        row["question"],
                                        row["student_draft"],
                                        temp)

        # *** true label from lookup ***
        gold = LABEL_LOOKUP.get(row["question"])
        if gold is None:
            print("!! question not found in label table"); continue

        if teacher_ans != gold:
            bad += 1
            print("!! teacher disagrees with gold label")
            # you may 'continue' here to drop bad rows from Track B

        # write Track A
        json.dump({"question": row["question"], "answer": gold}, A); A.write("\n")
        # write Track B
        full = row["question"] + "\n\nThought:\n" + cot
        json.dump({"question": full, "answer": gold}, B); B.write("\n")
        time.sleep(pause)

    A.close(); B.close()
    print(f"Mismatches: {bad}/{total}  =  {bad/total:.1%}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/student_drafts.jsonl")
    ap.add_argument("--out", default="data/strategyqa")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temp", type=float, default=0)
    ap.add_argument("--pause", type=float, default=0.5,
                    help="seconds to sleep between calls")
    main(**vars(ap.parse_args()))

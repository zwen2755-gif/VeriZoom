"""Scores of the open-ended runs from the generated texts, per method and budget, over the samples every method
finished, with the paired tests against a reference method (default H2O):
  grounding  mIoU and R@0.5 / R@0.7 of the 'Answer: S - E' interval with the ground-truth moment
  openqa     accuracy: the normalized ground-truth answer equals, or is a word of, the stated answer
  caption    ROUGE-L F1 of the description against the full-cache description (consistency) and the reference
and the KL / top-1 agreement over the replayed full-cache answer.
    python scripts/score_open.py runs/local/grounding_charades_7b [--json out.json]
"""
import argparse, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
parser.add_argument("--index", type=Path, default=None, help="index.jsonl of the task (default: from config.json)")
parser.add_argument("--reference", default="h2o")
parser.add_argument("--json", type=Path, default=None)
arguments = parser.parse_args()

config = json.loads((arguments.run / "config.json").read_text(encoding="utf-8"))
index = arguments.index or Path(config["data"]["root"]) / "index.jsonl"
entries = [json.loads(l) for l in index.read_text(encoding="utf-8").splitlines()]
WORD = re.compile(r"[a-z0-9]+")


def stated(text):
    """The stated answer; a generation that ends before its 'Answer:' line has none (scored as wrong, for every
    method alike), so that a long description cannot pass by mentioning the answer word somewhere."""
    found = re.findall(r"[Aa]nswer\s*:\s*(.+)", text)
    return found[-1].strip() if found else ""


def interval(text):
    numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", stated(text))]
    return (min(numbers[:2]), max(numbers[:2])) if len(numbers) >= 2 else None


def iou(a, b):
    if a is None:
        return 0.0
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


NUMBERS = dict(zip("0 1 2 3 4 5 6 7 8 9 10".split(),
                   "zero one two three four five six seven eight nine ten".split()))
ARTICLES = {"a", "an", "the"}


def words(text):
    return WORD.findall(text.lower())


def answer_words(text):
    """Words of a short answer with digits spelled out and articles dropped (MSVD-QA answers are single words)."""
    return [NUMBERS.get(w, w) for w in words(text) if w not in ARTICLES]


def rouge_l(candidate, reference):
    a, b = words(candidate), words(reference)
    if not a or not b:
        return 0.0
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b):
            current.append(previous[j] + 1 if x == y else max(previous[j + 1], current[j]))
        previous = current
    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


def openqa(text, answer):
    said, gold = answer_words(stated(text)), answer_words(str(answer))
    return float(bool(gold) and (said == gold or (len(gold) == 1 and gold[0] in said)))


records = defaultdict(dict)
for line in (arguments.run / "records.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    records[r["method"].split("@")[0], r["keep"]][r["sample"]] = r
most = max(len(v) for v in records.values())
records = {k: v for k, v in records.items() if len(v) >= 0.8 * most or k[0] == "full"}
complete = sorted(set.intersection(*(set(v) for v in records.values())))
full = records[("full", 1.0)]
task = next(iter(full.values()))["task"]
sample_entry = {}
for i, e in enumerate(entries):
    sample_entry[f"{e['task']}_{i:04d}"] = e


def score(record, key):
    e = sample_entry[key]
    if task == "grounding":
        value = iou(interval(record["text"]), e["reference"])
        return {"miou": value, "r5": float(value >= 0.5), "r7": float(value >= 0.7)}
    if task == "caption":
        return {"rouge_full": rouge_l(record["text"], full[key]["text"]), "rouge_ref": rouge_l(record["text"], e["reference"])}
    return {"acc": openqa(record["text"], e["reference"])}


print(f"## {arguments.run.name} ({task}): {len(complete)} samples finished by all {len(records)} configs\n")
out = {"task": task, "samples": len(complete), "rows": []}
for (method, keep), values in sorted(records.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    scores = [score(values[s], s) for s in complete]
    row = {"method": method, "keep": keep, **{m: float(np.mean([x[m] for x in scores])) for m in scores[0]}}
    if method != "full":
        kl = np.array([values[s]["kl"] for s in complete])
        row.update(kl=float(kl.mean()), agree=float(np.mean([values[s]["agree"] for s in complete])))
        ref = records.get((arguments.reference, keep))
        if ref is not None and method != arguments.reference:
            rkl = np.array([ref[s]["kl"] for s in complete])
            row.update(change=float(kl.mean() / rkl.mean() - 1), p=float(wilcoxon(kl, rkl).pvalue) if np.any(kl != rkl) else 1.0)
    out["rows"].append(row)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
if arguments.json:
    arguments.json.write_text(json.dumps(out, indent=1), encoding="utf-8")

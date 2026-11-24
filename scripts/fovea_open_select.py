"""Exits and read form of open-ended VeriZoom, chosen on the training split, then applied to the test sets.

The cascade returns a0 if c0 >= tau_g; else a1 if c1 >= tau_f and a1 agrees with a0; else a2 of the chosen read form.
Selection: highest accuracy on runs/fovea/mmerwtrain/open{tag}.jsonl, then the lowest latency.

    python scripts/fovea_open_select.py              # 3B
    python scripts/fovea_open_select.py --tag _7b    # 7B
"""
import argparse
import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fovea_open import answerable, norm  # noqa: E402

RUN = ROOT / "runs/fovea"
TESTS = ["vstar", "hrbench4k", "hrbench8k", "mmerw"]
GRID = [round(0.05 * i, 2) for i in range(21)] + [1.01]


def load(path):
    """rows of a run, without the label-taxonomy questions (fovea_open.answerable)"""
    rows = [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []
    return [r for r in rows if answerable(r)]


def cascade(r, tg, tf, form, rule="both"):
    """-> (correct, seconds, stage) for one row"""
    o = r["ours"]
    e = o["read"][rule]
    s = o["seconds"]
    t = s["glance"]
    if o["c0"] >= tg:
        return o["ok0"], t, 0
    t += s["propose"] + s["verify"]
    if e["c1"] >= tf and norm(e["a1"]) == norm(o["a0"]):
        return e["ok1"], t, 1
    return e[form]["ok2"], t + e[form]["seconds"], 2


def evaluate(rows, tg, tf, form, rule="both"):
    out = [cascade(r, tg, tf, form, rule) for r in rows]
    n = len(out)
    return (100 * sum(o[0] for o in out) / n, sum(o[1] for o in out) / n,
            [round(100 * sum(o[2] == k for o in out) / n) for k in range(3)], [bool(o[0]) for o in out])


def mcnemar(a, b):
    n10 = sum(x and not y for x, y in zip(a, b))
    n01 = sum(y and not x for x, y in zip(a, b))
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    train = load(RUN / "mmerwtrain" / f"open{args.tag}.jsonl")
    forms = list(train[0]["ours"]["read"]["both"].keys() & {"sep", "warp"})
    best = None
    for form in sorted(forms):
        for tg in GRID:
            for tf in GRID:
                acc, sec, _, _ = evaluate(train, tg, tf, form)
                key = (round(acc, 6), -sec)
                if best is None or key > best[0]:
                    best = (key, tg, tf, form)
    _, tg, tf, form = best
    cfg = {"tau_g": tg, "tau_f": tf, "form": form, "train_n": len(train)}
    acc, sec, ex, _ = evaluate(train, tg, tf, form)
    print(f"selected on {len(train)} training questions: {cfg}  train acc {acc:.1f} sec {sec:.2f} exits {ex}")
    json.dump(cfg, open(RUN / "mmerwtrain" / f"open{args.tag}_config.json", "w"), indent=1)

    summary = {}
    for d in TESTS:
        rows = load(RUN / d / f"open{args.tag}.jsonl")
        if not rows or "ours" not in rows[0]:
            continue
        ids = [r["id"] for r in rows]
        res = {}
        o = [r["ours"] for r in rows]
        res["glance"] = [x["ok0"] for x in o]
        res["verify"] = [x["read"]["both"]["ok1"] for x in o]
        res["verify, uniform"] = [x["read"]["uniform"]["ok1"] for x in o]
        for f in forms:
            res[f"read {f}"] = [x["read"]["both"][f]["ok2"] for x in o]
        cacc, csec, cex, ccor = evaluate(rows, tg, tf, form)
        res["VeriZoom"] = ccor
        secs = {"VeriZoom": csec}
        for k in rows[0]:
            if k.startswith("native") or k == "vicrop":
                res[k] = [r[k]["correct"] for r in rows]
                secs[k] = sum(r[k]["seconds"] for r in rows) / len(rows)
        for name, f in (("HiDe", "hide_open.jsonl"), ("AttWarp", "attwarp_open.jsonl")):
            other = {x["id"]: x for x in load(RUN / d / f)}
            if other and all(i in other for i in ids):
                res[name] = [other[i]["correct"] for i in ids]
                secs[name] = sum(other[i]["seconds"] for i in ids) / len(ids)
        print(f"\n{d}  n={len(rows)}  exits {cex}")
        for k, v in res.items():
            p = "" if k == "VeriZoom" else f"  p={mcnemar(res['VeriZoom'], v):.3g}"
            print(f"  {k:18s} {100 * sum(v) / len(v):5.1f}  {secs.get(k, float('nan')):6.2f}s{p}")
        summary[d] = {k: {"acc": 100 * sum(v) / len(v), "sec": secs.get(k), "correct": v} for k, v in res.items()}
        summary[d]["_exits"] = cex
    json.dump({"config": cfg, "sets": summary}, open(RUN / f"open_summary{args.tag}.json", "w"))


if __name__ == "__main__":
    main()

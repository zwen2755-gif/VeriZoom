"""Offline analysis of fovea_cascade.py runs.

select:  fit per-stage temperatures (NLL) and exit thresholds on a train run, print accuracy of every stage / rule /
         refine, and the accuracy-cost table of the cascade with exits.
apply:   apply a chosen config (rule, refine, T_g, T_f, tau_g, tau_f) to a test run.

usage: fovea_cascade_an.py select runs/fovea/mmerwtrain/cascade_sel.jsonl
       fovea_cascade_an.py apply runs/fovea/<d>/cascade_test.jsonl --config runs/fovea/mmerwtrain/cascade_config.json
"""
import argparse
import json
import math

import torch


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def temp(p, T):
    return torch.softmax(torch.log(torch.tensor(p).clamp_min(1e-12)) / T, -1)


def margin(p):
    s = p.sort(descending=True).values
    return float(s[0] - s[1]) if len(s) > 1 else 1.0


def fit_T(probs, answers):
    best = (1e9, 1.0)
    for T in [0.5 + 0.05 * i for i in range(51)]:
        nll = -sum(math.log(float(temp(p, T)[a]) + 1e-12) for p, a in zip(probs, answers)) / len(probs)
        best = min(best, (nll, T))
    return best[1]


def cascade(rows, rule, refine, Tg, Tf, tg, tf, consistency=True):
    """refine: read key (r0 / w0 / s0 ...) or an int (old runs: r<int>)"""
    key = refine if isinstance(refine, str) else f"r{refine}"
    """-> (accuracy, mean seconds, exit shares)"""
    correct, seconds, exits = 0, 0.0, [0, 0, 0]
    for r in rows:
        a = r["answer"]
        e = r["read"][rule]
        p0 = temp(r["p0"], Tg)
        p1 = temp(e["p1"], Tf)
        s = r["seconds"]
        t = s["glance"]
        if margin(p0) >= tg:
            pred, stage = int(p0.argmax()), 0
        else:
            t += s["propose"] + s["focus"]
            if margin(p1) >= tf and (not consistency or int(p1.argmax()) == int(p0.argmax())):
                pred, stage = int(p1.argmax()), 1
            else:
                t += e[key]["seconds"]
                pred, stage = int(torch.tensor(e[key]["probs"]).argmax()), 2
        correct += pred == a
        seconds += t
        exits[stage] += 1
    n = len(rows)
    return 100 * correct / n, seconds / n, [round(100 * x / n) for x in exits]


GRID = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]


def select(path):
    rows = load(path)
    n = len(rows)
    ans = [r["answer"] for r in rows]
    rules = list(rows[0]["read"])
    refines = sorted(k for k in rows[0]["read"][rules[0]] if k[0] in "rws" and k[1:].isdigit())
    acc = lambda preds: 100 * sum(int(p) == a for p, a in zip(preds, ans)) / n
    print(f"{path} n={n}")
    print(f"  glance p0            {acc([torch.tensor(r['p0']).argmax() for r in rows]):.1f}")
    print(f"  focus best single k  {acc([torch.tensor(r['P1'][int(torch.tensor(r['read'][rules[0]]['w']).argmax())]).argmax() for r in rows]):.1f}  (argmax w of {rules[0]})")
    print(f"  focus oracle any k   {100 * sum(any(int(torch.tensor(p).argmax()) == r['answer'] for p in r['P1']) for r in rows) / n:.1f}")
    for rule in rules:
        line = f"  {rule:8s} p1 {acc([torch.tensor(r['read'][rule]['p1']).argmax() for r in rows]):.1f}"
        for x in refines:
            line += f" | {x} {acc([torch.tensor(r['read'][rule][x]['probs']).argmax() for r in rows]):.1f}"
        line += f" | regions {sum(len(r['read'][rule]['chosen']) for r in rows) / n:.2f}"
        print(line)
    sec = {k: sum(r["seconds"][k] for r in rows) / n for k in ("glance", "propose", "focus")}
    print("  seconds " + " ".join(f"{k} {v:.2f}" for k, v in sec.items())
          + " | " + " ".join(f"{x} {sum(r['read'][rules[0]][x]['seconds'] for r in rows) / n:.2f}" for x in refines))
    Tg = fit_T([r["p0"] for r in rows], ans)
    best = None
    for rule in rules:
        Tf = fit_T([r["read"][rule]["p1"] for r in rows], ans)
        for x in refines:
            for tg in GRID:
                for tf in GRID:
                    a, s, e = cascade(rows, rule, x, Tg, Tf, tg, tf)
                    key = (round(a, 1), -s)
                    if best is None or key > best[0]:
                        best = (key, dict(rule=rule, refine=x, Tg=Tg, Tf=Tf, tau_g=tg, tau_f=tf), a, s, e)
    _, cfg, a, s, e = best
    print(f"  T_g {Tg:.2f}; best cascade {a:.1f} at {s:.2f} s, exits {e}: {cfg}")
    print("  accuracy-cost (best rule/refine, tau_f fixed, tau_g swept):")
    for tg in GRID:
        a2, s2, e2 = cascade(rows, cfg["rule"], cfg["refine"], Tg, cfg["Tf"], tg, cfg["tau_f"])
        print(f"    tau_g {tg:4.2f}: {a2:.1f}  {s2:.2f} s  exits {e2}")
    out = path.replace(".jsonl", "_config.json").replace("cascade_sel", "cascade")
    json.dump(cfg, open(out, "w"), indent=1)
    print("  ->", out)


def apply(path, cfg_path):
    rows = load(path)
    cfg = json.load(open(cfg_path))
    a, s, e = cascade(rows, cfg["rule"], cfg["refine"], cfg["Tg"], cfg["Tf"], cfg["tau_g"], cfg["tau_f"])
    print(f"{path} n={len(rows)} cascade {a:.1f} {s:.2f} s exits {e}")
    return a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["select", "apply"])
    parser.add_argument("path")
    parser.add_argument("--config")
    args = parser.parse_args()
    if args.mode == "select":
        select(args.path)
    else:
        apply(args.path, args.config)


if __name__ == "__main__":
    main()

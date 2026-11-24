"""Offline: which glance signals pick the view to read (K = 1, exact from the labels), per glance size and view set.

Glance signals come from fovea_glance_sweep.py (runs/fovea/<dataset>/glance_sweep.pt: question and generic-prompt
glances at size G); read-in labels and views from the label records (multi-scale after --extend-scales).
  vicrop       mean over the view's tokens of question / generic attention (ViCrop-style relative attention)
  attention    question attention (sum over layers)
  fq           forecaster on question-glance features (depth-binned attention share + K/V norms) + static
  fq+r         fq + depth-binned relative-attention features
Forecasters: 5-fold within the dataset (kf) and leave-one-dataset-out (lodo: fitted on the other datasets only).
View sets: fine (scale 1), le2 (scale <= 2), all.

usage: fovea_analyze_glance.py [--datasets vstar hrbench4k hrbench8k] [--sizes 256 1024]
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from fovea.views import View, depth_bins, token_owner, view_features  # noqa: E402
from fovea_analyze import ridge  # noqa: E402

SETS = {"fine": 1.0, "le2": 2.0, "all": 99.0}


def infogain(letters):
    probs = letters.float().clamp_min(1e-9)
    value = (probs * probs.log()).sum(-1)
    return (value - value.mean()) / (value.std() + 1e-6)


def glance_features(question, generic, owner, boxes, detail, bins, ratio=True) -> torch.Tensor:
    """View features from a question glance and a generic glance (per_token (L, 4, Ng) each): depth-binned attention
    share and K/V norms of the question glance, optionally the depth-binned relative attention, and static detail /
    area / centre. Shared by the offline analysis and fovea_e2e.py."""
    parts = [depth_bins(view_features(question, owner), bins).flatten(1)]
    if ratio:
        parts.append(depth_bins(view_features(question / (generic + 1e-6), owner)[..., :2], bins).flatten(1))
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    center = torch.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
    detail = (detail - detail.mean()) / (detail.std() + 1e-6)
    return torch.cat(parts + [detail[:, None], area[:, None], center], 1)


def correct_logprob(letters, correct):
    """Supervised label (training split only): standardized log-probability of the ground-truth letter per view."""
    probs = letters.float().clamp_min(1e-9)
    value = (probs / probs.sum(-1, keepdim=True))[:, correct].log()
    return (value - value.mean()) / (value.std() + 1e-6)


def build(record, glance, bins, keep, label="infogain"):
    views = [View(tuple(b), tuple(p)) for (b, p), k in zip(zip(record["views"], record["pixels"]), keep) if k]
    owner = token_owner(views, *glance["grid"])
    question, generic = glance["question"].float(), glance["generic"].float()
    boxes = [v.box for v in views]
    detail = record.get("detail_fast", record["detail"]).float()[keep]
    of = owner.float()
    ratio = (question[:, 0] / (generic[:, 0] + 1e-6)).mean(0)
    return {"fq": glance_features(question, generic, owner, boxes, detail, bins, ratio=False),
            "fq+r": glance_features(question, generic, owner, boxes, detail, bins, ratio=True),
            "vicrop": of @ ratio / of.sum(1).clamp_min(1),
            "attention": of @ question[:, 0].sum(0) / of.sum(1).clamp_min(1) ** 0.5,
            "label": (infogain(record["tile_letters"][keep]) if label == "infogain"
                      else correct_logprob(record["tile_letters"][keep], record["correct"])),
            "ok": record["tile_letters"][keep].argmax(1) == record["correct"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["vstar", "hrbench4k", "hrbench8k"])
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--bins", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--save", default="", help="fit one forecaster on all --datasets (fine views, --sizes[0], "
                                                   "--key) and save it to this path")
    parser.add_argument("--key", default="fq+r", choices=["fq", "fq+r"])
    parser.add_argument("--label", default="infogain", choices=["infogain", "correct"],
                        help="infogain (self-supervised) or correct (log-prob of the ground-truth letter)")
    parser.add_argument("--layers", type=int, default=0, help="use only the first N decoder layers of the glance")
    args = parser.parse_args()
    data = {}
    for name in args.datasets:
        run = ROOT / f"runs/fovea/{name}"
        records = [torch.load(p) for p in sorted((run / "labels").glob("*.pt"))]
        sweep = torch.load(run / "glance_sweep.pt")
        if args.layers:
            sweep = {size: {i: {**g, "question": g["question"][:args.layers], "generic": g["generic"][:args.layers]}
                            for i, g in glances.items()} for size, glances in sweep.items()}
        data[name] = (records, sweep)
    if args.save:
        size = args.sizes[0]
        items = []
        for name, (records, sweep) in data.items():
            for r in records:
                keep = r.get("scale", torch.ones(len(r["views"]))) <= 1
                items.append(build(r, sweep[size][r["id"]], args.bins, keep, args.label))
        x, y = torch.cat([it[args.key] for it in items]), torch.cat([it["label"] for it in items])
        mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
        z = torch.cat([(x - mean) / std, torch.ones(len(x), 1)], 1)
        penalty = args.alpha * torch.eye(z.shape[1])
        penalty[-1, -1] = 0
        w = torch.linalg.solve(z.T @ z + penalty, z.T @ y)
        torch.save({"mean": mean, "std": std, "w": w, "bins": args.bins, "glance": size, "ratio": args.key == "fq+r",
                    "scales": [1.0], "datasets": args.datasets, "views": len(x), "layers": args.layers or None,
                    "label": args.label}, ROOT / args.save)
        print(f"saved {args.save}: {x.shape[1]} features, {len(x)} views from {args.datasets}")
        return
    print(f"{'dataset':10s} {'G':>5s} {'views':>5s} {'random':>7s} {'any':>7s} {'vicrop':>7s} {'attn':>6s} "
          f"{'fq kf':>6s} {'fq+r kf':>7s} {'fq lodo':>7s} {'fq+r lodo':>9s}")
    for size in args.sizes:
        for subset, limit in SETS.items():
            built = {}
            for name, (records, sweep) in data.items():
                built[name] = []
                for r in records:
                    scale = r.get("scale", torch.ones(len(r["views"])))
                    built[name].append(build(r, sweep[size][r["id"]], args.bins, scale <= limit, args.label))
            for name, items in built.items():
                acc = lambda scores: 100 * sum(bool(it["ok"][int(s.argmax())]) for it, s in zip(items, scores)) / len(items)  # noqa: E731
                row = {"random": 100 * sum(float(it["ok"].float().mean()) for it in items) / len(items),
                       "any": 100 * sum(bool(it["ok"].any()) for it in items) / len(items),
                       "vicrop": acc([it["vicrop"] for it in items]), "attn": acc([it["attention"] for it in items])}
                for key in ("fq", "fq+r"):
                    preds = [None] * len(items)
                    for fold in range(5):
                        train = [i for i in range(len(items)) if i % 5 != fold]
                        model = ridge(torch.cat([items[i][key] for i in train]),
                                      torch.cat([items[i]["label"] for i in train]), args.alpha)
                        for i in range(fold, len(items), 5):
                            preds[i] = model(items[i][key])
                    row[f"{key} kf"] = acc(preds)
                    others = [it for other, its in built.items() if other != name for it in its]
                    model = ridge(torch.cat([it[key] for it in others]), torch.cat([it["label"] for it in others]),
                                  args.alpha)
                    row[f"{key} lodo"] = acc([model(it[key]) for it in items])
                views = sum(len(it["label"]) for it in items) / len(items)
                print(f"{name:10s} {size:5d} {subset:>5s}:{views:<4.0f} {row['random']:7.1f} {row['any']:7.1f} "
                      f"{row['vicrop']:7.1f} {row['attn']:6.1f} {row['fq kf']:6.1f} {row['fq+r kf']:7.1f} "
                      f"{row['fq lodo']:7.1f} {row['fq+r lodo']:9.1f}")


if __name__ == "__main__":
    main()

"""Pilot P0, phase 2 (offline): how well do different scores pick the view to read?

Uses the labels of fovea_label.py. For K = 1 (glance + one tile at native resolution) the accuracy of any scorer is
exact offline, since every tile's answer distribution is stored. Scorers:
  random     expected accuracy over tiles
  attention  question attention on the glance tokens of the tile (summed over layers, mean over heads)
  detail     high-frequency content the glance cannot show
  forecast   ridge regression on depth-binned glance features + static features, k-fold over samples
  oracle     the tile with the lowest KL to the reference (read-in label)
  target     the tile that covers the most of the annotated target box
Also reports the ranking quality (Spearman with the read-in label) and how often the chosen tile covers the target.

usage: fovea_analyze.py --dataset vstar [--bins 7] [--folds 5]
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fovea.views import View, depth_bins, token_owner, view_features  # noqa: E402


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra, rb = a.argsort().argsort().float(), b.argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm() + 1e-9))


def owner_of(record) -> torch.Tensor:
    """(V, glance tokens) bool: glance tokens inside each view (space for images, time for videos)."""
    if "glance_times" in record:
        groups, rows, cols = record["glance_grid"]
        times = record["glance_times"]
        centres = torch.tensor([(times[2 * g] + times[2 * g + 1]) / 2 for g in range(groups)]).repeat_interleave(rows * cols)
        masks = []
        for start, _, end, _ in record["pixels"]:
            mask = (centres >= start) & (centres < end)
            if not mask.any():
                mask[(centres - (start + end) / 2).abs().argmin()] = True
            masks.append(mask)
        return torch.stack(masks)
    rows, cols = record["glance_grid"]
    views = [View(tuple(b), tuple(p)) for b, p in zip(record["views"], record["pixels"])]
    return token_owner(views, rows, cols)


def features_of(record, bins: int) -> torch.Tensor:
    owner = owner_of(record)
    per_view = view_features(record["per_token"].float(), owner)          # (V, L, 4)
    binned = depth_bins(per_view, bins).flatten(1)                         # (V, bins*4)
    boxes = torch.tensor(record["views"], dtype=torch.float32)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    center = torch.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
    detail = record.get("detail_fast", record["detail"]).float()   # detail_fast (fovea_detail_fast.py) when present
    detail = (detail - detail.mean()) / (detail.std() + 1e-6)
    return torch.cat([binned, detail[:, None], area[:, None], center], 1)


LABEL = "infogain"


def label_of(record) -> torch.Tensor:
    """Read-in value of every view, standardized within the sample.
    infogain:  information gain of reading the view, i.e. how much the entropy of the answer distribution drops
               (ranked within the sample, so the glance entropy cancels); needs neither the ground truth nor a
               reference answer.
    certainty: how certain the model becomes after reading the view (max answer probability).
    reference: how close the answer moves to the model's own answer on the full input (-log KL)."""
    if LABEL == "infogain":
        probs = record["tile_letters"].float().clamp_min(1e-9)
        value = (probs * probs.log()).sum(-1)
    elif LABEL == "certainty":
        value = record["tile_letters"].float().max(1).values
    else:
        value = -torch.log(record["tile_kl"].float().clamp_min(1e-8))
    return (value - value.mean()) / (value.std() + 1e-6)


def ridge(x: torch.Tensor, y: torch.Tensor, alpha: float):
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    z = torch.cat([(x - mean) / std, torch.ones(len(x), 1)], 1)
    penalty = alpha * torch.eye(z.shape[1])
    penalty[-1, -1] = 0
    w = torch.linalg.solve(z.T @ z + penalty, z.T @ y)
    return lambda q: torch.cat([(q - mean) / std, torch.ones(len(q), 1)], 1) @ w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="vstar")
    parser.add_argument("--bins", type=int, default=7)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--label", default="infogain", choices=["infogain", "certainty", "reference"])
    parser.add_argument("--train", default="", help="fit on this dataset's labels instead of k-fold (zero-shot)")
    parser.add_argument("--save-model", action="store_true",
                        help="fit on all labels of --dataset and save runs/fovea/<dataset>/forecaster.pt")
    args = parser.parse_args()
    global LABEL
    LABEL = args.label
    # "a+b" pools the labels of several datasets (a forecaster fitted once on all of them)
    records = [torch.load(p) for name in args.dataset.split("+")
               for p in sorted((ROOT / f"runs/fovea/{name}/labels").glob("*.pt"))]
    (ROOT / f"runs/fovea/{args.dataset}").mkdir(parents=True, exist_ok=True)
    print(f"{len(records)} samples")
    feats = [features_of(r, args.bins) for r in records]
    labels = [label_of(r) for r in records]

    if args.save_model:
        x, y = torch.cat(feats), torch.cat(labels)
        mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
        z = torch.cat([(x - mean) / std, torch.ones(len(x), 1)], 1)
        penalty = args.alpha * torch.eye(z.shape[1])
        penalty[-1, -1] = 0
        w = torch.linalg.solve(z.T @ z + penalty, z.T @ y)
        torch.save({"mean": mean, "std": std, "w": w, "bins": args.bins, "alpha": args.alpha, "label": args.label,
                    "samples": len(records)}, ROOT / f"runs/fovea/{args.dataset}/forecaster.pt")
        print(f"saved forecaster ({x.shape[1]} features, {len(x)} views)")

    forecasts = [None] * len(records)
    if args.train:
        source = [torch.load(p) for p in sorted((ROOT / f"runs/fovea/{args.train}/labels").glob("*.pt"))]
        model = ridge(torch.cat([features_of(r, args.bins) for r in source]), torch.cat([label_of(r) for r in source]),
                      args.alpha)
        forecasts = [model(f) for f in feats]
    for fold in range(0 if args.train else args.folds):
        train = [i for i in range(len(records)) if i % args.folds != fold]
        model = ridge(torch.cat([feats[i] for i in train]), torch.cat([labels[i] for i in train]), args.alpha)
        for i in range(fold, len(records), args.folds):
            forecasts[i] = model(feats[i])

    def attention_score(r):
        owner = owner_of(r).float()
        per_token = r["per_token"].float()[:, 0].sum(0)                  # (Ng) mean-over-heads, summed over layers
        return owner @ per_token / owner.sum(1).clamp_min(1) ** 0.5      # between sum and mean: favours dense views

    scorers = {
        "attention": [attention_score(r) for r in records],
        "detail": [r["detail"].float() for r in records],
        "forecast": forecasts,
        "oracle": [-r["tile_kl"].float() for r in records],
        "certainty": [r["tile_letters"].float().max(1).values for r in records],
        "target": [r["coverage"].float() for r in records],
    }
    torch.save({r["id"]: {k: v[i] for k, v in scorers.items()} for i, r in enumerate(records)},
               ROOT / f"runs/fovea/{args.dataset}/scores{'_from_' + args.train if args.train else ''}.pt")
    correct = lambda r, i: int(r["tile_letters"][i].argmax()) == r["correct"]  # noqa: E731
    ref_acc = sum(int(r["ref_letters"].argmax()) == r["correct"] for r in records) / len(records)
    glance_acc = sum(int(r["glance_letters"].argmax()) == r["correct"] for r in records) / len(records)
    random_acc = sum(sum(correct(r, i) for i in range(len(r["tile_kl"]))) / len(r["tile_kl"]) for r in records)
    print(f"reference (native, ~{sum(r['ref_tokens'] for r in records) / len(records):.0f} tok): {100 * ref_acc:.1f}")
    print(f"glance only (~{sum(r['glance_tokens'] for r in records) / len(records):.0f} tok): {100 * glance_acc:.1f}")
    print(f"glance + random tile: {100 * random_acc / len(records):.1f}")
    has_box = [r for r in records if r["coverage"].max() > 0]
    print(f"{'scorer':10s} {'acc':>6s} {'spearman':>9s} {'target-hit':>10s}  (per category acc)")
    for name, scores in scorers.items():
        hits, rho, cover, by_cat = 0, [], 0, {}
        for r, s in zip(records, scores):
            pick = int(s.argmax())
            ok = correct(r, pick)
            hits += ok
            by_cat.setdefault(r["category"], []).append(ok)
            rho.append(spearman(s, label_of(r)))
            cover += float(r["coverage"][pick] >= 0.5 * r["coverage"].max()) if r["coverage"].max() > 0 else 0
        cats = " ".join(f"{k[:8]}={100 * sum(v) / len(v):.1f}" for k, v in by_cat.items())
        print(f"{name:10s} {100 * hits / len(records):6.1f} {sum(rho) / len(rho):9.3f} "
              f"{100 * cover / max(1, len(has_box)):10.1f}  {cats}")


if __name__ == "__main__":
    main()

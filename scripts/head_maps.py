"""Training-split localization score of every (layer, head), as in fovea_rank_heads.py (kind q), for the figure:
saves runs/fovea/mmerwtrain/head_map_<tag>.pt with the (L, H) accuracy matrix."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
run = ROOT / "runs/fovea/mmerwtrain"
for sweep, tag in (("head_sweep.pt", "3b"), ("head_sweep_7b.pt", "7b")):
    heads = torch.load(run / sweep)
    layer_ids = heads.pop("_layer_ids", None)
    total, n = None, 0
    for p in sorted((run / "labels").glob("*.pt")):
        r = torch.load(p)
        if r["id"] not in heads:
            continue
        h = heads[r["id"]]
        keep = r.get("scale", torch.ones(len(r["views"]))) <= 1
        ok = (r["tile_letters"][keep].argmax(1) == r["correct"]).float()
        q = h["q_sum"].float() + 1e-9
        s = q / q.mean(0, keepdim=True)
        hit = ok[s.argmax(0)]
        total = hit if total is None else total + hit
        n += 1
    acc = 100 * total / n
    H = acc.shape[1]
    top = acc.flatten().argsort(descending=True)[:3]
    print(tag, n, tuple(acc.shape), "layers", layer_ids[:3] if layer_ids else None,
          [(int(i) // H, int(i) % H, round(float(acc.flatten()[i]), 1)) for i in top],
          "median", round(float(acc.median()), 1))
    torch.save({"acc": acc, "layer_ids": layer_ids, "n": n}, run / f"head_map_{tag}.pt")

"""Head selection on the train split only (MME-RealWorld train, no boxes): rank every (layer, head) by the K=1 view
accuracy of its question-attention share (the view whose glance tokens get the largest share of the head's question
attention, scored by the per-view correctness labels in runs/fovea/mmerwtrain/labels), print the top --top as
"layer:head ..." for fovea_cascade.py --heads.

usage: fovea_rank_heads.py [--sweep head_sweep.pt] [--top 3] [--kind q]
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", default="head_sweep.pt")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--kind", default="q", choices=["q", "r"])
    args = parser.parse_args()
    run = ROOT / "runs/fovea/mmerwtrain"
    heads = torch.load(run / args.sweep)
    layer_ids = heads.pop("_layer_ids", None)   # absolute decoder layers when only some layers attend
    total, n = None, 0
    for p in sorted((run / "labels").glob("*.pt")):
        r = torch.load(p)
        if r["id"] not in heads:
            continue
        h = heads[r["id"]]
        keep = r.get("scale", torch.ones(len(r["views"]))) <= 1
        ok = (r["tile_letters"][keep].argmax(1) == r["correct"]).float()
        q, g = h["q_sum"].float() + 1e-9, h["g_sum"].float() + 1e-9          # (V, L, H)
        s = q / q.mean(0, keepdim=True) if args.kind == "q" else (q / g) / (q / g).mean(0, keepdim=True)
        hit = ok[s.argmax(0)]                                                 # (L, H)
        total = hit if total is None else total + hit
        n += 1
    acc = 100 * total / n
    H = acc.shape[1]
    top = acc.flatten().argsort(descending=True)[:args.top]
    print(f"{n} train questions; top {args.top}: "
          + ", ".join(f"L{int(i) // H}H{int(i) % H} {float(acc.flatten()[i]):.1f}" for i in top), file=sys.stderr)
    layer = (lambda i: layer_ids[i]) if layer_ids else (lambda i: i)
    print(" ".join(f"{layer(int(i) // H)}:{int(i) % H}" for i in top))


if __name__ == "__main__":
    main()

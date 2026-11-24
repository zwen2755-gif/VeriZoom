"""Diagnostic: is the location of the evidence visible in the glance signals, and at which glance resolution?

For each glance size G, runs the question glance (and a generic-description glance, for the attention ratio of
ViCrop) and stores per-tile features. Then reports how often the top tile covers the annotated target for:
  attention   question attention summed over layers
  ratio       question attention / generic-description attention (per layer, then summed)
  located     ridge on depth-binned features + ratio, trained k-fold to predict target coverage (an upper bound on
              what a linear read-out of the glance can locate; it uses the annotation, so it is diagnostic only)

usage: fovea_glance_sweep.py --dataset vstar --sizes 256 576 1024 2304
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
Image.MAX_IMAGE_PIXELS = None

import fovea  # noqa: E402,F401
from vzcore.models import load_model  # noqa: E402
from fovea.canvas import Piece, resized  # noqa: E402
from fovea.views import GlanceRecorder, View, depth_bins, token_owner, view_features  # noqa: E402
from fovea_analyze import ridge  # noqa: E402
from fovea_label import LOADERS, Reader, question_text  # noqa: E402

GENERIC = "Write a general description of the image."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="vstar")
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 576, 1024, 2304])
    parser.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    args = parser.parse_args()
    loader, data_root = LOADERS[args.dataset]
    samples = {s["id"]: s for s in loader(ROOT / data_root)}
    run = ROOT / f"runs/fovea/{args.dataset}"
    records = [torch.load(p) for p in sorted((run / "labels").glob("*.pt"))]
    cache = run / "glance_sweep.pt"
    store = torch.load(cache) if cache.exists() else {}
    adapter = load_model("qwen2_5_vl", model_id=str(ROOT / args.model), dtype="bfloat16")
    reader = Reader(adapter)

    def glance(image, text):
        def make(prompt, spans):
            last = int(spans[-1].max())
            return GlanceRecorder(spans[0], torch.arange(last + 1, prompt.length, device=prompt.embeds.device))
        _, _, recorder = reader.log_probs([Piece(image)], text, make)
        return recorder.per_token().cpu().float()

    for size in args.sizes:
        todo = [r for r in records if r["id"] not in store.get(size, {})]  # only records added since the last sweep
        if not todo:
            continue
        store.setdefault(size, {})
        for r in todo:
            sample = samples[r["id"]]
            image = Image.open(sample["image"]).convert("RGB")
            small = resized(image, size)
            text, _, _ = question_text(sample)
            store[size][r["id"]] = {"question": glance(small, text).half(), "generic": glance(small, GENERIC).half(),
                                    "grid": (small.height // 28, small.width // 28)}
            torch.cuda.empty_cache()
        torch.save(store, cache)
        print(f"glance {size} done", flush=True)

    if not any(r["coverage"].max() > 0 for r in records):
        return  # no target boxes (HR-Bench): the cache is the product
    print(f"{'G':>5s} {'attention':>9s} {'ratio':>7s} {'located':>8s}   (top tile covers >= half of the best cover)")
    with_box = [r for r in records if r["coverage"].max() > 0]
    for size in sorted(store):
        feats, targets, attn_hits, ratio_hits = [], [], 0, 0
        for r in with_box:
            g = store[size][r["id"]]
            views = [View(tuple(b), tuple(p)) for b, p in zip(r["views"], r["pixels"])]
            owner = token_owner(views, *g["grid"]).float()
            question, generic = g["question"].float(), g["generic"].float()
            attention = owner @ question[:, 0].sum(0)
            ratio_tok = (question[:, 0] / (generic[:, 0] + 1e-6)).mean(0)
            ratio = (owner @ ratio_tok) / owner.sum(1).clamp_min(1)
            good = r["coverage"] >= 0.5 * r["coverage"].max()
            attn_hits += bool(good[attention.argmax()])
            ratio_hits += bool(good[ratio.argmax()])
            f = depth_bins(view_features(question, owner.bool()), 7).flatten(1)
            fr = depth_bins(view_features(question / (generic + 1e-6), owner.bool()), 7)[..., :2].flatten(1)
            feats.append(torch.cat([f, fr], 1))
            targets.append(r["coverage"].float())
        located = 0
        for fold in range(5):
            train = [i for i in range(len(feats)) if i % 5 != fold]
            model = ridge(torch.cat([feats[i] for i in train]), torch.cat([targets[i] for i in train]), 10.0)
            for i in range(fold, len(feats), 5):
                cover = targets[i]
                located += bool(cover[model(feats[i]).argmax()] >= 0.5 * cover.max())
        n = len(with_box)
        print(f"{size:5d} {100 * attn_hits / n:9.1f} {100 * ratio_hits / n:7.1f} {100 * located / n:8.1f}")


if __name__ == "__main__":
    main()

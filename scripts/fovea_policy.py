"""Pilot P0, phase 3 (online): accuracy, tokens and latency of sampling policies at matched visual-token budgets.

  downscale-B     the whole image resized to B tokens (B matched to each glance+K policy, and the reference cap)
  <scorer>-K      glance + the K best tiles of a scorer (non-maximum suppression on overlapping tiles), read at
                  native resolution; scorers come from fovea_analyze.py (scores*.pt)

usage: fovea_policy.py --dataset vstar --scores scores.pt [--k 1 2 4] [--scorers attention forecast oracle]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
Image.MAX_IMAGE_PIXELS = None

import fovea  # noqa: E402,F401  (fast prefill kernel)
from vzcore.models import load_model  # noqa: E402
from fovea.canvas import Piece, resized  # noqa: E402
from fovea_label import LOADERS, Reader, kl, question_text  # noqa: E402


def overlap(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / max(1e-9, smaller)


def pick(scores: torch.Tensor, boxes, k: int, limit: float = 0.5) -> list[int]:
    chosen = []
    for index in scores.argsort(descending=True).tolist():
        if all(overlap(boxes[index], boxes[j]) < limit for j in chosen):
            chosen.append(index)
        if len(chosen) == k:
            break
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="vstar")
    parser.add_argument("--scores", default="scores.pt")
    parser.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--k", type=int, nargs="*", default=[1, 2, 4])
    parser.add_argument("--scorers", nargs="*", default=["attention", "forecast", "oracle", "random"])
    parser.add_argument("--glance-tokens", type=int, default=256)
    parser.add_argument("--tile-tokens", type=int, default=576)
    parser.add_argument("--ref-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--faithful", action="store_true", help="coordinate-faithful positions, no crop captions")
    parser.add_argument("--verify", type=int, nargs="*", default=[], help="propose-and-verify with M proposals")
    parser.add_argument("--verify-scorer", default="forecast")
    args = parser.parse_args()

    loader, data_root = LOADERS[args.dataset]
    samples = {s["id"]: s for s in loader(ROOT / data_root)}
    run = ROOT / f"runs/fovea/{args.dataset}"
    scores = torch.load(run / args.scores)
    labels = {p.stem.replace("__", "/"): p for p in (run / "labels").glob("*.pt")}
    ids = [i for i in sorted(scores) if i in samples and i in labels]
    if args.limit:
        ids = ids[:args.limit]
    out = run / f"policy{args.tag}.jsonl"
    done = set()
    if out.exists():
        done = {(r["id"], r["method"]) for r in map(json.loads, out.read_text().splitlines())}

    adapter = load_model("qwen2_5_vl", model_id=str(ROOT / args.model), dtype="bfloat16")
    reader = Reader(adapter)
    reader.canvas.faithful = args.faithful
    tokenizer = adapter.processor.tokenizer

    with out.open("a") as sink:
        for number, sid in enumerate(ids):
            sample, record = samples[sid], torch.load(labels[sid])
            image = Image.open(sample["image"]).convert("RGB")
            text, correct, letters = question_text(sample)
            letter_ids = torch.tensor([tokenizer.convert_tokens_to_ids(l) for l in letters], device=adapter.device)
            boxes, pixels = record["views"], record["pixels"]
            glance_image = resized(image, args.glance_tokens)
            native = (image.height / 28, image.width / 28)
            ref = None

            def evaluate(method, pieces):
                nonlocal ref
                if (sid, method) in done and method != "reference":
                    return
                torch.cuda.synchronize()
                start = time.perf_counter()
                lp, prompt, _ = reader.log_probs(pieces, text)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                if method == "reference":
                    ref = lp
                    if (sid, method) in done:
                        return
                letter = int(lp[letter_ids].argmax())
                row = {"id": sid, "category": sample["category"], "method": method, "tokens": prompt.num_visual,
                       "correct": letter == letters.index(correct), "kl": kl(ref, lp), "seconds": seconds}
                sink.write(json.dumps(row) + "\n")

            evaluate("reference", [Piece(resized(image, args.ref_tokens))])
            budgets = set()
            for k in args.k:
                for name in args.scorers:
                    if name == "random":
                        chosen = torch.randperm(len(boxes), generator=torch.Generator().manual_seed(number))
                        chosen = pick(-chosen.argsort().float(), boxes, k)
                    else:
                        chosen = pick(scores[sid][name].float(), boxes, k)
                    pieces = [Piece(glance_image, "" if args.faithful else "Overview of the whole image:",
                                    {"box": (0, 0, 1, 1), "grid": native})]
                    for index in sorted(chosen):
                        crop = resized(image.crop(tuple(pixels[index])), args.tile_tokens)
                        x0, y0, x1, y1 = (round(v * 1000) for v in boxes[index])
                        caption = "" if args.faithful else (f"Zoomed-in view of region [{x0}, {y0}, {x1}, {y1}] "
                                                            "(0-1000 coordinates):")
                        pieces.append(Piece(crop, caption, {"box": tuple(boxes[index]), "grid": native}))
                    budgets.add(sum(p.tokens for p in pieces))
                    evaluate(f"{name}-{k}", pieces)
            for budget in sorted({args.glance_tokens + k * args.tile_tokens for k in args.k}):
                evaluate(f"downscale-{budget}", [Piece(resized(image, budget))])
            # propose-and-verify: the forecaster proposes M views, each is read once with the glance, and the answer
            # comes from the read with the lowest answer entropy; cost = all M reads
            for m in args.verify:
                method = f"verify-{m}"
                if (sid, method) in done:
                    continue
                chosen = pick(scores[sid][args.verify_scorer].float(), boxes, m)
                best, tokens, seconds = None, 0, 0.0
                for index in chosen:
                    crop = resized(image.crop(tuple(pixels[index])), args.tile_tokens)
                    x0, y0, x1, y1 = (round(v * 1000) for v in boxes[index])
                    pieces = [Piece(glance_image, "Overview of the whole image:"),
                              Piece(crop, f"Zoomed-in view of region [{x0}, {y0}, {x1}, {y1}] (0-1000 coordinates):")]
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    lp, prompt, _ = reader.log_probs(pieces, text)
                    torch.cuda.synchronize()
                    seconds += time.perf_counter() - start
                    tokens += prompt.num_visual
                    probs = lp[letter_ids].softmax(0)
                    entropy = float(-(probs * probs.clamp_min(1e-9).log()).sum())
                    if best is None or entropy < best[0]:
                        best = (entropy, lp, index)
                row = {"id": sid, "category": sample["category"], "method": method, "tokens": tokens,
                       "correct": int(best[1][letter_ids].argmax()) == letters.index(correct),
                       "kl": kl(ref, best[1]), "seconds": seconds, "view": best[2]}
                sink.write(json.dumps(row) + "\n")
            sink.flush()
            torch.cuda.empty_cache()
            print(f"[{number + 1}/{len(ids)}] {sid}", flush=True)


if __name__ == "__main__":
    main()

"""Diagnostic (read stage): with the target located perfectly (V* ground-truth boxes), how much does the FORM in which
the zoomed evidence is given to the model matter? Every variant answers the same question; only the read differs.

  overview      the 1024-token image alone (= the glance answer)
  ov+tile       overview + the fine grid tile that holds the targets (current FOVEA read), caption with coordinates
  crop          the targets' union box (padded to >= 2x and >= 336 px) alone, read at ~576 tokens (upsampled if small)
  ov+crop       overview + that crop, caption with coordinates
  ov+crop-pos   overview + that crop, no captions, every token at its source position in the overview's frame
  ov+each       overview + one crop per target (two-target questions), captions
  ov+each-pos   the same, position-aligned, no captions

usage: fovea_diag_readform.py [--limit N]   -> runs/fovea/vstar/readform.jsonl (resumable)
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import fovea  # noqa: E402,F401
from vzcore.models import load_model  # noqa: E402
from fovea.canvas import PATCH, Piece, resized  # noqa: E402
from fovea.views import tile_views  # noqa: E402
from fovea_label import LOADERS, Reader, question_text  # noqa: E402

OVERVIEW = "Overview of the whole image:"


def to_tokens(image, tokens):
    """`image` resized (up or down) to about `tokens` LLM tokens, sides multiples of PATCH."""
    scale = (tokens / ((image.width / PATCH) * (image.height / PATCH))) ** 0.5
    w = max(PATCH, round(image.width * scale / PATCH) * PATCH)
    h = max(PATCH, round(image.height * scale / PATCH) * PATCH)
    while (w // PATCH) * (h // PATCH) > tokens:
        w, h = (w - PATCH, h) if w >= h else (w, h - PATCH)
    return image.resize((w, h), Image.BICUBIC)


def padded(box, width, height, factor=2.0, minimum=336):
    """(x0, y0, x1, y1) of `box` = (x, y, w, h) grown to factor x its size and at least `minimum` px a side."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    sw, sh = min(width, max(w * factor, minimum)), min(height, max(h * factor, minimum))
    x0, y0 = min(max(cx - sw / 2, 0), width - sw), min(max(cy - sh / 2, 0), height - sh)
    return int(x0), int(y0), int(x0 + sw), int(y0 + sh)


def union(boxes):
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return x0, y0, x1 - x0, y1 - y0


def zoom(image, region, tokens, faithful):
    x0, y0, x1, y1 = region
    W, H = image.size
    box = (x0 / W, y0 / H, x1 / W, y1 / H)
    caption = "" if faithful else (f"Zoomed-in view of region [{', '.join(str(round(v * 1000)) for v in box)}] "
                                   "(0-1000 coordinates):")
    return Piece(to_tokens(image.crop(region), tokens), caption,
                 meta={"box": box, "grid": (H / PATCH, W / PATCH)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=576)
    args = parser.parse_args()
    loader, data_root = LOADERS["vstar"]
    samples = loader(ROOT / data_root)
    if args.limit:
        samples = samples[:args.limit]
    out = ROOT / "runs/fovea/vstar/readform.jsonl"
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    adapter = load_model("qwen2_5_vl", model_id=str(ROOT / "models/Qwen2.5-VL-3B-Instruct"), dtype="bfloat16")
    reader = Reader(adapter)
    tokenizer = adapter.processor.tokenizer
    with out.open("a") as sink:
        for number, sample in enumerate(samples):
            if sample["id"] in done:
                continue
            image = Image.open(sample["image"]).convert("RGB")
            image.load()
            W, H = image.size
            text, correct, letters = question_text(sample)
            letter_ids = torch.tensor([tokenizer.convert_tokens_to_ids(l) for l in letters], device=adapter.device)
            answer = letters.index(correct)
            boxes = sample["bboxes"]
            overview = resized(image, 1024)
            ov_meta = {"box": (0.0, 0.0, 1.0, 1.0), "grid": (H / PATCH, W / PATCH)}
            ub = union(boxes)
            tiles = tile_views(W, H, args.tokens, 0.25)
            inside = [v for v in tiles if v.pixels[0] <= ub[0] and v.pixels[1] <= ub[1]
                      and v.pixels[2] >= ub[0] + ub[2] and v.pixels[3] >= ub[1] + ub[3]]
            if not inside:   # no tile holds every target: the tile holding the largest one
                big = max(boxes, key=lambda b: b[2] * b[3])
                inside = [v for v in tiles if v.pixels[0] <= big[0] and v.pixels[1] <= big[1]
                          and v.pixels[2] >= big[0] + big[2] and v.pixels[3] >= big[1] + big[3]] or tiles[:1]
            tile = inside[0]
            region = padded(ub, W, H)
            variants = {
                "overview": [Piece(overview)],
                "ov+tile": [Piece(overview, OVERVIEW), zoom(image, tile.pixels, args.tokens, False)],
                "crop": [zoom(image, region, args.tokens, False)],
                "ov+crop": [Piece(overview, OVERVIEW), zoom(image, region, args.tokens, False)],
                "ov+crop-pos": [Piece(overview, meta=ov_meta), zoom(image, region, args.tokens, True)],
            }
            if len(boxes) > 1:
                each = [padded(b, W, H) for b in boxes]
                share = args.tokens // len(boxes)
                variants["ov+each"] = [Piece(overview, OVERVIEW)] + [zoom(image, r, share, False) for r in each]
                variants["ov+each-pos"] = [Piece(overview, meta=ov_meta)] + [zoom(image, r, share, True) for r in each]
            row = {"id": sample["id"], "category": sample["category"], "targets": len(boxes),
                   "union_side": max(ub[2], ub[3]), "region": region}
            for name, pieces in variants.items():
                reader.canvas.faithful = name.endswith("-pos")
                lp, prompt, _ = reader.log_probs(pieces, text)
                probs = lp[letter_ids].softmax(0)
                row[name] = {"correct": int(probs.argmax()) == answer, "tokens": prompt.num_visual,
                             "probs": probs.tolist()}
            reader.canvas.faithful = False
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            print(f"[{number + 1}/{len(samples)}] {sample['id']} "
                  + " ".join(f"{k}={'ok' if v['correct'] else 'x'}" for k, v in row.items() if isinstance(v, dict)),
                  flush=True)


if __name__ == "__main__":
    main()

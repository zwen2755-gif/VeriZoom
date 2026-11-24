"""Diagnostic (read form for several regions, ground-truth boxes on V*): how should several regions be read together
so that their spatial relation survives?

forms (every one with the same target boxes, padded as in fovea_diag_readform):
  sep        glance + one crop per target (caption with 0-1000 box), --tokens split equally   (current cascade read)
  union      glance + the union crop of all targets at --tokens
  warp       glance + order-preserving separable warp at --tokens: along x and y independently, a piecewise-linear
             monotone map that scales target intervals by s and the gaps between them by rho * s (one s for both
             axes, so targets keep their aspect; left/right/above/below order is preserved exactly)
  warp-solo  the warp alone at 1024 + --tokens tokens (the glance's budget folded in, context at rho * s)
  som        glance with numbered coloured boxes drawn on it + crops captioned with the same number and colour

usage: fovea_diag_layout.py [--tokens 576] [--rho 0.25] [--limit N]  -> runs/fovea/vstar/layout.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import fovea  # noqa: E402,F401
from vzcore.models import load_model  # noqa: E402
from fovea.canvas import PATCH, Piece, resized  # noqa: E402
from fovea_diag_readform import padded, to_tokens  # noqa: E402
from fovea_label import LOADERS, Reader, question_text  # noqa: E402

COLORS = [("red", (230, 30, 30)), ("blue", (30, 80, 230)), ("green", (20, 170, 40)), ("yellow", (240, 200, 0))]


def caption_of(box, W, H):
    b = tuple(round(v * 1000) for v in (box[0] / W, box[1] / H, box[2] / W, box[3] / H))
    return f"Zoomed-in view of region [{', '.join(map(str, b))}] (0-1000 coordinates):"


def intervals(lo_hi, length):
    """[(a, b, focus)] covering [0, length): focus where some [lo, hi) covers it"""
    cuts = sorted({0, length, *[v for ab in lo_hi for v in ab]})
    out = []
    for a, b in zip(cuts, cuts[1:]):
        if b > a:
            out.append((a, b, any(lo <= a and b <= hi for lo, hi in lo_hi)))
    return out


def warp(image, boxes, tokens, rho):
    """order-preserving separable piecewise-linear warp of `image`: boxes at scale s, gaps at rho * s, ~tokens"""
    W, H = image.size
    xs = intervals([(b[0], b[2]) for b in boxes], W)
    ys = intervals([(b[1], b[3]) for b in boxes], H)
    ex = sum((b - a) * (1 if f else rho) for a, b, f in xs)
    ey = sum((b - a) * (1 if f else rho) for a, b, f in ys)
    s = (tokens * PATCH * PATCH / (ex * ey)) ** 0.5
    widths = [max(1, round((b - a) * s * (1 if f else rho))) for a, b, f in xs]
    heights = [max(1, round((b - a) * s * (1 if f else rho))) for a, b, f in ys]
    out = Image.new("RGB", (sum(widths), sum(heights)))
    oy = 0
    for (ya, yb, _), h in zip(ys, heights):
        ox = 0
        for (xa, xb, _), w in zip(xs, widths):
            out.paste(image.crop((xa, ya, xb, yb)).resize((w, h), Image.BICUBIC), (ox, oy))
            ox += w
        oy += h
    return to_tokens(out, tokens)


def union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def marked(small, boxes, W, H):
    im = small.copy()
    draw = ImageDraw.Draw(im)
    sx, sy = small.width / W, small.height / H
    for j, b in enumerate(boxes):
        name, rgb = COLORS[j % len(COLORS)]
        x0, y0, x1, y1 = b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy
        draw.rectangle((x0, y0, x1, y1), outline=rgb, width=3)
        draw.rectangle((x0, max(0, y0 - 16), x0 + 14, max(16, y0)), fill=rgb)
        draw.text((x0 + 3, max(0, y0 - 15)), str(j + 1), fill=(255, 255, 255))
    return im


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=576)
    parser.add_argument("--rho", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    loader, data_root = LOADERS["vstar"]
    samples = loader(ROOT / data_root)
    if args.limit:
        samples = samples[:args.limit]
    adapter = load_model("qwen2_5_vl", model_id=str(ROOT / "models/Qwen2.5-VL-3B-Instruct"), dtype="bfloat16")
    reader = Reader(adapter)
    tok = adapter.processor.tokenizer
    out = ROOT / f"runs/fovea/vstar/layout{args.tag}.jsonl"
    with out.open("w") as sink:
        for number, sample in enumerate(samples):
            image = Image.open(sample["image"]).convert("RGB")
            image.load()
            W, H = image.size
            text, correct, letters = question_text(sample)
            letter_ids = torch.tensor([tok.convert_tokens_to_ids(l) for l in letters], device=adapter.device)
            answer = letters.index(correct)
            boxes = [padded(b, W, H) for b in sample["bboxes"]]
            small = resized(image, 1024)
            n = len(boxes)
            each = args.tokens // n
            forms = {
                "sep": [Piece(small)] + [Piece(to_tokens(image.crop(b), each), caption_of(b, W, H)) for b in boxes],
                "union": [Piece(small), Piece(to_tokens(image.crop(union(boxes)), args.tokens),
                                              caption_of(union(boxes), W, H))],
                "warp": [Piece(small), Piece(warp(image, boxes, args.tokens, args.rho),
                                             "Magnified view of the whole image (regions of interest enlarged, "
                                             "layout preserved):")],
                "warp-solo": [Piece(warp(image, boxes, 1024 + args.tokens, args.rho))],
                "som": [Piece(marked(small, boxes, W, H))]
                       + [Piece(to_tokens(image.crop(b), each),
                                f"Zoomed-in view of box {j + 1} ({COLORS[j % 4][0]}):") for j, b in enumerate(boxes)],
            }
            row = {"id": sample["id"], "category": sample["category"], "targets": n}
            for name, pieces in forms.items():
                lp, prompt, _ = reader.log_probs(pieces, text)
                probs = lp[letter_ids].softmax(0)
                row[name] = {"correct": int(probs.argmax()) == answer, "tokens": prompt.num_visual,
                             "probs": probs.tolist()}
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            print(f"[{number + 1}/{len(samples)}] {sample['id']} "
                  + " ".join(f"{k}={'o' if row[k]['correct'] else 'x'}" for k in forms), flush=True)


if __name__ == "__main__":
    main()

"""Evidence region from grounding words (prototype): each content word of the question (stop words removed, the
options and the instruction excluded) points, through a few localization heads of the 1024-token question glance,
at one glance token (the peak of its attention). Points that agree are merged (weighted by peak attention), weak
ones dropped; the region is the box around the kept points, padded to at least `--min-side` px a side (attention
gives where, not how large). The region is read alone, upsampled to ~--tokens tokens, and after the overview.

Uses runs/fovea/<dataset>/rows.pt (fovea_diag_rows.py) for the per-word attention.

usage: fovea_region.py --dataset vstar [--tag ""] [--limit N]   -> runs/fovea/<dataset>/region<tag>.jsonl
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
from fovea.canvas import Piece, resized  # noqa: E402
from fovea_diag_readform import OVERVIEW, to_tokens  # noqa: E402
from fovea_label import LOADERS, Reader, question_text  # noqa: E402

STOP = set("a an the is are was were be of on in at to from with by for and or what which who whose where when how "
           "why this that these those it its there their color colour side left right top bottom above below "
           "relative position located does do did can could would should image picture photo shown visible kind "
           "type".split())


def content_rows(tokens):
    """indices of the question's content words: up to the first line break (options follow), no stop words."""
    rows = []
    for i, t in enumerate(tokens[1:], 1):
        if "\n" in t:
            break
        w = t.strip().lower().strip("?.,")
        if w and w.isalpha() and w not in STOP:
            rows.append(i)
    return rows


def grounding_points(rec, heads, merge=2, keep=0.35):
    """[(row, col, weight)] on the glance grid."""
    R, C = rec["grid"]
    maps = rec["maps"][heads].float()
    m = (maps / maps.sum(-1, keepdim=True)).mean(0)                   # (rows, Ng)
    clusters = []
    for i in content_rows(rec["tokens"]):
        r, c = divmod(int(m[i].argmax()), C)
        w = float(m[i].max())
        for cl in clusters:
            if abs(cl[0] - r) <= merge and abs(cl[1] - c) <= merge:
                cl[2] += w
                break
        else:
            clusters.append([r, c, w])
    if not clusters:
        return []
    top = max(cl[2] for cl in clusters)
    return [tuple(cl) for cl in clusters if cl[2] >= keep * top]


def region_of(points, grid, size, min_side):
    R, C = grid
    W, H = size
    xs = [(c + 0.5) / C * W for _, c, _ in points]
    ys = [(r + 0.5) / R * H for r, _, _ in points]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    half = min_side / 2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sw, sh = min(W, max(x1 - x0 + min_side, 2 * half)), min(H, max(y1 - y0 + min_side, 2 * half))
    x0, y0 = min(max(cx - sw / 2, 0), W - sw), min(max(cy - sh / 2, 0), H - sh)
    return int(x0), int(y0), int(x0 + sw), int(y0 + sh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="vstar")
    parser.add_argument("--heads", type=int, nargs="+", default=[0, 1, 2], help="indices into fovea_diag_rows.HEADS")
    parser.add_argument("--min-side", type=int, default=336)
    parser.add_argument("--tokens", type=int, default=576)
    parser.add_argument("--keep", type=float, default=0.35)
    parser.add_argument("--tag", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    rows = torch.load(ROOT / f"runs/fovea/{args.dataset}/rows.pt")
    loader, data_root = LOADERS[args.dataset]
    samples = loader(ROOT / data_root)
    if args.limit:
        samples = samples[:args.limit]
    out = ROOT / f"runs/fovea/{args.dataset}/region{args.tag}.jsonl"
    adapter = load_model("qwen2_5_vl", model_id=str(ROOT / "models/Qwen2.5-VL-3B-Instruct"), dtype="bfloat16")
    reader = Reader(adapter)
    tokenizer = adapter.processor.tokenizer
    with out.open("w") as sink:
        for number, sample in enumerate(samples):
            rec = rows[sample["id"]]
            image = Image.open(sample["image"]).convert("RGB")
            image.load()
            W, H = image.size
            text, correct, letters = question_text(sample)
            letter_ids = torch.tensor([tokenizer.convert_tokens_to_ids(l) for l in letters], device=adapter.device)
            answer = letters.index(correct)
            points = grounding_points(rec, args.heads, keep=args.keep)
            row = {"id": sample["id"], "category": sample["category"], "points": points}
            if points:
                region = region_of(points, rec["grid"], (W, H), args.min_side)
                box = tuple(round(v * 1000) for v in (region[0] / W, region[1] / H, region[2] / W, region[3] / H))
                crop = to_tokens(image.crop(region), args.tokens)
                caption = f"Zoomed-in view of region [{', '.join(map(str, box))}] (0-1000 coordinates):"
                row["region"] = region
                for name, pieces in (("crop", [Piece(crop, caption)]),
                                     ("ov+crop", [Piece(resized(image, 1024), OVERVIEW), Piece(crop, caption)])):
                    lp, prompt, _ = reader.log_probs(pieces, text)
                    probs = lp[letter_ids].softmax(0)
                    row[name] = {"correct": int(probs.argmax()) == answer, "tokens": prompt.num_visual,
                                 "probs": probs.tolist()}
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            print(f"[{number + 1}/{len(samples)}] {sample['id']} points={len(points)} "
                  + " ".join(f"{k}={'ok' if v['correct'] else 'x'}" for k, v in row.items() if isinstance(v, dict)),
                  flush=True)


if __name__ == "__main__":
    main()

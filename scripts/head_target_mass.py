"""Where do the attention heads look on V*? For every head of the model, the attention of the question's content words
over the 1024-token glance (exactly as the glance of fovea_cascade.py: same prompt, same word rows, each word map
normalized over the glance tokens, then averaged over the words), and the share of that attention that falls inside
the annotated target boxes of V* (union of the targets, cells weighted by their overlap with the boxes). The target's
share of the image area is the share a uniform attention would give.

Saves runs/fovea/vstar/head_target_mass{tag}.pt = {"mass": (Q, L, H), "area": (Q,), "ids": [...], "maps": {id:
(L*H, R, C) fp16}, "L", "H"}. Analysis only: the heads of VeriZoom are selected on the MME-RealWorld training split.

usage: head_target_mass.py --model models/Qwen2.5-VL-3B-Instruct --layers 36 --heads 16 [--offload ...] --tag _3b
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fovea_cascade as FC  # noqa: E402
from fovea_cascade import PATCH, Heads, Piece, Reader, load_model, question_text, resized, word_rows, LOADERS  # noqa: E402


def overlap_grid(boxes, W, H, R, C):
    """(R, C) fraction of each glance cell covered by the union of the boxes (x, y, w, h) in image pixels"""
    ys = torch.linspace(0, H, R + 1)
    xs = torch.linspace(0, W, C + 1)
    cover = torch.zeros(R, C)
    for b in boxes:                           # exact rectangle overlap; V* targets of one question do not overlap
        oy = (torch.minimum(ys[1:], torch.tensor(float(b[1] + b[3]))) - torch.maximum(ys[:-1], torch.tensor(float(b[1])))).clamp_min(0)
        ox = (torch.minimum(xs[1:], torch.tensor(float(b[0] + b[2]))) - torch.maximum(xs[:-1], torch.tensor(float(b[0])))).clamp_min(0)
        cover += oy[:, None] * ox[None, :] / ((H / R) * (W / C))
    return cover.clamp(max=1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--offload", nargs="*", default=[])
    ap.add_argument("--tag", default="_3b")
    args = ap.parse_args()
    FC.HEADS[:] = [(l, h) for l in range(args.layers) for h in range(args.heads)]
    FC.vz_attention.LOWER_RIGHT_CAUSAL = True
    adapter = load_model("qwen2_5_vl", model_id=str(ROOT / args.model), dtype="bfloat16",
                         **({"offload": args.offload} if args.offload else {}))
    reader = Reader(adapter)
    tok = adapter.processor.tokenizer
    samples = LOADERS["vstar"][0](ROOT / LOADERS["vstar"][1])
    masses, areas, ids, maps = [], [], [], {}
    for n, s in enumerate(samples):
        image = Image.open(s["image"]).convert("RGB")
        W, H = image.size
        text, _, _ = question_text(s)
        small = resized(image, 1024)
        R, C = small.height // PATCH, small.width // PATCH
        prompt, spans = reader.canvas.build([Piece(small)], text)
        cut = int(spans[0].max()) + 1
        rec = Heads(torch.tensor(word_rows(prompt, cut - 1, tok), device=adapter.device), spans[0])
        cache = reader.decoder.new_cache(1, prompt.length + 1)
        reader.decoder.forward(prompt.embeds, prompt.token_index, prompt.positions, cache, rec, "prefill")
        m = rec.maps()                                            # (heads, words, S)
        m = m / m.sum(-1, keepdim=True).clamp_min(1e-12)
        m = m.mean(1).view(len(FC.HEADS), R, C).cpu()             # (heads, R, C), each sums to 1
        cover = overlap_grid(s["bboxes"], W, H, R, C)
        masses.append((m * cover).sum((1, 2)).view(args.layers, args.heads))
        areas.append(float(cover.mean()))
        ids.append(s["id"])
        maps[s["id"]] = m.half()
        reader.canvas.cache.clear()
        print(f"[{n + 1}/{len(samples)}] {s['id']} target {100 * areas[-1]:.2f}% of the image, best head "
              f"{100 * float(masses[-1].max()):.1f}%", flush=True)
    out = ROOT / "runs/fovea/vstar" / f"head_target_mass{args.tag}.pt"
    torch.save({"mass": torch.stack(masses), "area": torch.tensor(areas), "ids": ids, "maps": maps,
                "L": args.layers, "H": args.heads}, out)
    print("->", out)


if __name__ == "__main__":
    main()

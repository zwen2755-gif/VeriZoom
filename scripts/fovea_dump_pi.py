"""Dump the proposal map pi of some questions (glance stage of fovea_cascade.py, same heads as the test run) for the
figures. Checks that the boxes from each dumped map equal the stored boxes of runs/fovea/<set>/cascade_test_h.

    python scripts/fovea_dump_pi.py vstar:direct_attributes/sa_25835 hrbench8k:single/364
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fovea_cascade as FC  # noqa: E402
from fovea_cascade import (HEADS, PATCH, Heads, Piece, Reader, clip_box, load_model, mass_box, modes,  # noqa: E402
                           question_text, resized, word_maps, word_rows, LOADERS)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="*", default=["vstar:direct_attributes/sa_25835"], help="dataset:id")
    ap.add_argument("--heads", nargs="*", default=["24:5", "22:10", "22:0"])
    ap.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--tag", default="_test_h", help="cascade run whose stored boxes are checked")
    args = ap.parse_args()
    HEADS[:] = [tuple(map(int, x.split(":"))) for x in args.heads]
    FC.vz_attention.LOWER_RIGHT_CAUSAL = True
    adapter = load_model("qwen2_5_vl", model_id=str(ROOT / args.model), dtype="bfloat16")
    reader = Reader(adapter)
    tok = adapter.processor.tokenizer
    out = ROOT / "runs/fovea/figure"
    out.mkdir(parents=True, exist_ok=True)
    loaded = {}
    for item in args.items:
        ds, qid = item.split(":", 1)
        if ds not in loaded:
            loaded[ds] = {s["id"]: s for s in LOADERS[ds][0](ROOT / LOADERS[ds][1])}
        sample = loaded[ds][qid]
        image = Image.open(sample["image"]).convert("RGB")
        W, H = image.size
        text, _, _ = question_text(sample)
        small = resized(image, 1024)
        R, C = small.height // PATCH, small.width // PATCH
        gprompt, gspans = reader.canvas.build([Piece(small)], text)
        cut = int(gspans[0].max()) + 1
        rec = Heads(torch.tensor(word_rows(gprompt, cut - 1, tok), device=adapter.device), gspans[0])
        cache = reader.decoder.new_cache(1, gprompt.length + 1)
        reader.decoder.forward(gprompt.embeds, gprompt.token_index, gprompt.positions, cache, rec, "prefill")
        pi = word_maps(rec).sum(0).view(R, C)
        pi = (pi / pi.sum()).cpu()
        boxes = []
        for r, c in modes(pi, 4, 3):
            r0, r1, c0, c1 = mass_box(pi, r, c, 0.8, 3)
            boxes.append(list(clip_box(c0 / C * W, r0 / R * H, (c1 + 1) / C * W, (r1 + 1) / R * H, W, H, 336)))
        path = ROOT / f"runs/fovea/{ds}/cascade{args.tag}.jsonl"
        stored = None
        if path.exists():
            for line in open(path):
                row = json.loads(line)
                if row["id"] == qid:
                    stored = row["boxes"]
                    break
        name = f"pi_{ds}_{qid.replace('/', '__')}.pt"
        torch.save({"pi": pi, "boxes": boxes}, out / name)   # boxes are checked against the local run if absent here
        print(item, "boxes match stored" if stored == boxes else f"BOXES DIFFER {boxes} vs {stored}", "->", name,
              flush=True)


if __name__ == "__main__":
    main()

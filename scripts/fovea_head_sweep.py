"""Diagnostic (glance signal): per-head question / generic attention of the 1024 glance, pooled per fine view.

Stores runs/fovea/<dataset>/head_sweep.pt = {id: {"q_sum", "q_max", "g_sum", "g_max": (V, L, H) fp16, "q_tok": (L, H, Ng)
fp16 for V* only}}: sum and max over the view's glance tokens of the attention each head gives from the question
rows. Same glance as fovea_glance_sweep.py (resized to --size tokens, same prompts). Resumable.

usage: fovea_head_sweep.py --dataset vstar [--size 1024]
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
from fovea.canvas import PATCH, Piece, resized  # noqa: E402
from fovea.views import GlanceRecorder, View, token_owner  # noqa: E402
from fovea_glance_sweep import GENERIC  # noqa: E402
from fovea_label import LOADERS, Reader, question_text  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="vstar")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--offload", nargs="*", default=[])
    parser.add_argument("--out", default="head_sweep.pt")
    args = parser.parse_args()
    loader, data_root = LOADERS[args.dataset]
    samples = {s["id"]: s for s in loader(ROOT / data_root)}
    run = ROOT / f"runs/fovea/{args.dataset}"
    cache = run / args.out
    store = torch.load(cache) if cache.exists() else {}
    paths = [p for p in sorted((run / "labels").glob("*.pt"))]
    adapter = load_model("qwen2_5_vl", model_id=str(ROOT / args.model), dtype="bfloat16",
                         **({"offload": args.offload} if args.offload else {}))
    reader = Reader(adapter)

    def glance(image, text):
        def make(prompt, spans):
            last = int(spans[-1].max())
            return GlanceRecorder(spans[0], torch.arange(last + 1, prompt.length, device=prompt.embeds.device),
                                  per_head=True)
        with torch.no_grad():
            _, _, recorder = reader.log_probs([Piece(image)], text, make)
        glance.layer_ids = recorder.layer_ids    # decoder layer of each recorded row (attention layers)
        return recorder.per_head().float().cpu()  # (L, H, Ng)

    added = 0
    for number, path in enumerate(paths):
        record = torch.load(path)
        if record["id"] in store:
            continue
        sample = samples[record["id"]]
        image = Image.open(sample["image"]).convert("RGB")
        small = resized(image, args.size)
        text, _, _ = question_text(sample)
        keep = record.get("scale", torch.ones(len(record["views"]))) <= 1
        views = [View(tuple(b), tuple(p)) for (b, p), k in zip(zip(record["views"], record["pixels"]), keep) if k]
        owner = token_owner(views, small.height // PATCH, small.width // PATCH).float()  # (V, Ng)
        entry = {}
        for key, prompt_text in (("q", text), ("g", GENERIC)):
            heads = glance(small, prompt_text)
            entry[f"{key}_sum"] = torch.einsum("lhn,vn->vlh", heads, owner).half()
            masked = heads[None] * owner[:, None, None] - (1 - owner[:, None, None])  # non-members -> -1
            entry[f"{key}_max"] = masked.amax(-1).half()
            if args.dataset == "vstar":
                entry[f"{key}_tok"] = heads.half()
        entry["grid"] = (small.height // PATCH, small.width // PATCH)
        store[record["id"]] = entry
        store["_layer_ids"] = list(glance.layer_ids)
        added += 1
        torch.cuda.empty_cache()
        if added % 200 == 0:
            torch.save(store, cache)
            print(f"{args.dataset}: {number + 1}/{len(paths)}", flush=True)
    torch.save(store, cache)
    print(f"{args.dataset}: head sweep done ({len(store)} questions)", flush=True)


if __name__ == "__main__":
    main()

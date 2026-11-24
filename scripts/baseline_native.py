"""Baseline: the whole image read once at each --tokens budget (the native image when the budget exceeds it).

8192 is the native reference of fovea_e2e.py; smaller budgets give the downscale-to-budget baseline at matched cost.

usage: baseline_native.py --dataset vstar [--tokens 1600 8192] [--model ... --offload embed head] [--tag _3b]
       -> runs/fovea/<dataset>/native<tag>.jsonl
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

import fovea  # noqa: E402,F401
from vzcore.models import load_model  # noqa: E402
from fovea.canvas import Piece, resized  # noqa: E402
from fovea_label import LOADERS, Reader, question_text  # noqa: E402


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="vstar", choices=sorted(LOADERS))
    parser.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--offload", nargs="*", default=[])
    parser.add_argument("--tokens", type=int, nargs="+", default=[1600, 8192])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--every", type=int, default=1, help="every n-th sample (latency subsets)")
    parser.add_argument("--warm", action="store_true", help="run the first sample once untimed before the loop")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    loader, data_root = LOADERS[args.dataset]
    samples = loader(ROOT / data_root)
    if args.limit:
        samples = samples[:args.limit]
    samples = samples[::args.every]
    out = ROOT / f"runs/fovea/{args.dataset}/native{args.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    adapter = load_model("qwen2_5_vl", model_id=str(ROOT / args.model), dtype="bfloat16",
                         **({"offload": args.offload} if args.offload else {}))
    reader = Reader(adapter)
    tok = adapter.processor.tokenizer
    hits = {t: 0 for t in args.tokens}
    # resume: the file is written in sample order, so its first rows are the first samples
    old = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
    assert all(r["id"] == s["id"] for r, s in zip(old, samples)), "existing rows do not match the samples"
    for r in old:
        for t in args.tokens:
            hits[t] += r[str(t)]["correct"]
    warm = args.warm and not old           # warm-up: the first sample runs twice, the first run is not written
    if warm:
        samples = samples[:1] + samples
    with out.open("a") as sink:
        for number, sample in enumerate(samples):
            if number < len(old):
                continue
            image = Image.open(sample["image"]).convert("RGB")
            image.load()
            text, correct, letters = question_text(sample)
            letter_ids = torch.tensor([tok.convert_tokens_to_ids(l) for l in letters], device=adapter.device)
            answer = letters.index(correct)
            row = {"id": sample["id"], "category": sample["category"], "answer": answer}
            for t in args.tokens:
                torch.cuda.synchronize()
                start = time.perf_counter()
                lp, prompt, _ = reader.log_probs([Piece(resized(image, t))], text)
                probs = lp[letter_ids].softmax(0)
                torch.cuda.synchronize()
                ok = int(probs.argmax()) == answer
                hits[t] += ok
                row[str(t)] = {"correct": ok, "tokens": prompt.num_visual, "seconds": time.perf_counter() - start,
                               "probs": probs.tolist()}
                reader.canvas.cache.clear()
                torch.cuda.empty_cache()
            if warm and number == 0:
                hits = {t: 0 for t in args.tokens}
                continue
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            print(f"[{number + 1}/{len(samples)}] {sample['id']} "
                  + " ".join(f"{t}:{100 * hits[t] / (number + 1):.1f}" for t in args.tokens), flush=True)


if __name__ == "__main__":
    main()

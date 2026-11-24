"""Pilot P0, phase 1: glance features and read-in labels of every view of a high-resolution image.

For each multiple-choice sample:
  reference  the whole image at native resolution (capped at --ref-tokens), the model's own answer distribution
  glance     the whole image at --glance-tokens; records per-layer question attention and K/V norms of its tokens
  read-in    for every tile view: glance + that tile at native resolution (<= --tile-tokens), KL to the reference
             and probability of the correct option
One .pt per sample under runs/fovea/<dataset>/labels, so the run resumes.

usage: fovea_label.py --dataset vstar [--limit N]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
Image.MAX_IMAGE_PIXELS = None

from vzcore.engine.forward import Decoder  # noqa: E402
from vzcore.models import load_model  # noqa: E402
from fovea.canvas import PATCH, Canvas, Piece, resized  # noqa: E402
from fovea.views import GlanceRecorder, detail_loss, tile_views  # noqa: E402

LETTERS = "ABCDEFGH"
INSTRUCTION = "Answer with the option's letter from the given choices directly."


def load_vstar(root: Path):
    samples = []
    for category in ("direct_attributes", "relative_position"):
        for meta in sorted((root / category).glob("*.json")):
            info = json.loads(meta.read_text())
            image = next(p for p in meta.parent.glob(meta.stem + ".*") if p.suffix.lower() != ".json")
            samples.append({"id": f"{category}/{meta.stem}", "image": image, "question": info["question"],
                            "options": info["options"], "answer": 0, "bboxes": info.get("bbox", []),
                            "category": category})
    return samples


def load_hrbench(root: Path, variant: str = "4k", cycles: tuple[int, ...] = (0,)):
    """HR-Bench: 200 questions x 4 option rotations (cycle_category). Images are decoded once into
    data_local/hrbench<variant>/images. Options keep the benchmark's own order."""
    import base64

    import pyarrow.parquet as pq
    images = ROOT / f"data_local/hrbench{variant}/images"
    images.mkdir(parents=True, exist_ok=True)
    samples = []
    table = pq.ParquetFile(root / f"hr_bench_{variant}.parquet")
    for batch in table.iter_batches(batch_size=16):
        for row in batch.to_pylist():
            if row["cycle_category"] not in cycles:
                continue
            path = images / f"{row['index']}.jpg"
            if not path.exists():
                data = row["image"]
                if len(data) < 64:  # rotations of a question refer to the image of its first row by index
                    continue
                path.write_bytes(base64.b64decode(data))
            samples.append({"id": f"{row['category']}/{row['index']}", "image": path, "question": row["question"],
                            "options": [row[k] for k in "ABCD"], "answer": "ABCD".index(row["answer"]),
                            "bboxes": [], "category": row["category"], "keep_order": True})
    return samples


def load_mmerw(root: Path):
    """MME-RealWorld-Lite: 1919 five-option questions on high-resolution real-world images; options keep their order
    (the "(A) " prefixes are stripped and re-lettered identically)."""
    import ast
    import re
    rows = json.loads((root / "data/MME-RealWorld-Lite.json").read_text(encoding="utf-8"))
    samples = []
    for r in rows:
        choices = r["Answer choices"]
        choices = ast.literal_eval(choices) if isinstance(choices, str) else choices
        options = [re.sub(r"^\([A-E]\)\s*", "", c) for c in choices]
        samples.append({"id": r["Question_id"], "image": root / "data/imgs" / r["Image"], "question": r["Text"],
                        "options": options, "answer": "ABCDE".index(r["Ground truth"].strip()[0]), "bboxes": [],
                        "category": f"{r['Subtask']}/{r['Task']}", "keep_order": True})
    return samples


def load_mmerw_train(root: Path):
    """MME-RealWorld training split (train_split.json: Lite-disjoint questions and images, one question per image,
    stratified like Lite by subtask x task); images extracted by scripts/extract_mmerw.py. Questions whose image is
    not extracted yet are skipped."""
    import ast
    import re
    rows = json.loads((root / "train_split.json").read_text(encoding="utf-8"))
    samples = []
    for r in rows:
        image = root / "imgs" / r["Image"]
        if not image.exists():
            continue
        choices = r["Answer choices"]
        choices = ast.literal_eval(choices) if isinstance(choices, str) else choices
        options = [re.sub(r"^\([A-F]\)\s*", "", c) for c in choices]
        samples.append({"id": r["Question_id"], "image": image, "question": r["Text"], "options": options,
                        "answer": "ABCDEF".index(r["Ground truth"].strip()[0]), "bboxes": [],
                        "category": f"{r['Subtask']}/{r['Task']}", "keep_order": True})
    return samples


LOADERS = {"vstar": (load_vstar, "data_local/raw/vstar"),
           "mmerw": (load_mmerw, "data_local/raw/mmerw_lite"),
           "mmerwtrain": (load_mmerw_train, "data_local/raw/mmerw_full"),
           "hrbench4k": (lambda r: load_hrbench(r, "4k"), "data_local/raw/hrbench"),
           "hrbench8k": (lambda r: load_hrbench(r, "8k"), "data_local/raw/hrbench"),
           # cycle 0 alone always has the answer at A; the official protocol averages all four option rotations
           "hrbench4k_c123": (lambda r: load_hrbench(r, "4k", (1, 2, 3)), "data_local/raw/hrbench"),
           "hrbench8k_c123": (lambda r: load_hrbench(r, "8k", (1, 2, 3)), "data_local/raw/hrbench")}


def question_text(sample) -> tuple[str, str, list[str]]:
    order = list(range(len(sample["options"])))
    if not sample.get("keep_order"):
        random.Random(sample["id"]).shuffle(order)  # fixed per sample: no position bias towards the first option
    letters = LETTERS[:len(order)]
    lines = [f"{letters[i]}. {sample['options'][j]}" for i, j in enumerate(order)]
    correct = letters[order.index(sample["answer"])]
    return f"{sample['question']}\n" + "\n".join(lines) + f"\n{INSTRUCTION}", correct, list(letters)


def coverage(view, bboxes) -> float:
    """Largest fraction of a target box (x, y, w, h in pixels) inside the view."""
    best = 0.0
    x0, y0, x1, y1 = view.pixels
    for bx, by, bw, bh in bboxes:
        ix = max(0, min(x1, bx + bw) - max(x0, bx))
        iy = max(0, min(y1, by + bh) - max(y0, by))
        best = max(best, ix * iy / max(1, bw * bh))
    return best


class Reader:
    def __init__(self, adapter):
        self.adapter, self.canvas, self.decoder = adapter, Canvas(adapter), Decoder(adapter)

    @torch.no_grad()
    def log_probs(self, pieces, text, recorder_factory=None):
        prompt, spans = self.canvas.build(pieces, text)
        recorder = recorder_factory(prompt, spans) if recorder_factory else None
        cache = self.decoder.new_cache(1, prompt.length + 1)
        logits = self.decoder.forward(prompt.embeds, prompt.token_index, prompt.positions, cache, recorder, "prefill")
        return F.log_softmax(logits[0, -1], -1), prompt, recorder


def kl(reference, other) -> float:
    return float((reference.exp() * (reference - other)).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="vstar")
    parser.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--ref-tokens", type=int, default=8192)
    parser.add_argument("--glance-tokens", type=int, default=256)
    parser.add_argument("--tile-tokens", type=int, default=576)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="")
    parser.add_argument("--extend-scales", type=float, nargs="*", default=[],
                        help="append coarse views (footprint x scale, read at --tile-tokens) to existing labels")
    args = parser.parse_args()

    loader, data_root = LOADERS[args.dataset]
    samples = loader(ROOT / data_root)
    if args.limit:
        samples = samples[:args.limit]
    out = ROOT / (args.out or f"runs/fovea/{args.dataset}/labels")
    out.mkdir(parents=True, exist_ok=True)
    adapter = load_model("qwen2_5_vl", model_id=str(ROOT / args.model), dtype="bfloat16")
    reader = Reader(adapter)
    tokenizer = adapter.processor.tokenizer

    for number, sample in enumerate(samples):
        target = out / (sample["id"].replace("/", "__") + ".pt")
        if args.extend_scales:
            if target.exists():
                extend(target, sample, args, reader, tokenizer, number, len(samples))
            continue
        if target.exists():
            continue
        start = time.perf_counter()
        image = Image.open(sample["image"]).convert("RGB")
        text, correct, letters = question_text(sample)
        letter_ids = torch.tensor([tokenizer.convert_tokens_to_ids(l) for l in letters], device=adapter.device)

        reference_image = resized(image, args.ref_tokens)
        ref, ref_prompt, _ = reader.log_probs([Piece(reference_image)], text)

        glance_image = resized(image, args.glance_tokens)
        rows, cols = glance_image.height // PATCH, glance_image.width // PATCH

        def make_recorder(prompt, spans):
            last = int(spans[-1].max())
            return GlanceRecorder(spans[0], torch.arange(last + 1, prompt.length, device=prompt.embeds.device))

        glance, glance_prompt, recorder = reader.log_probs([Piece(glance_image, "Overview of the whole image:")],
                                                           text, make_recorder)
        views = tile_views(image.width, image.height, args.tile_tokens, args.overlap)
        glance_scale = glance_image.width / image.width
        tile_kl, tile_letters, details, covers = [], [], [], []
        for view in views:
            crop = resized(view.crop(image), args.tile_tokens)
            x0, y0, x1, y1 = (round(v * 1000) for v in view.box)
            pieces = [Piece(glance_image, "Overview of the whole image:"),
                      Piece(crop, f"Zoomed-in view of region [{x0}, {y0}, {x1}, {y1}] (0-1000 coordinates):")]
            lp, _, _ = reader.log_probs(pieces, text)
            tile_kl.append(kl(ref, lp))
            tile_letters.append(lp[letter_ids].softmax(0).cpu())
            details.append(detail_loss(image, view, glance_scale))
            covers.append(coverage(view, sample["bboxes"]))

        record = {
            "id": sample["id"], "category": sample["category"], "size": image.size, "correct": letters.index(correct),
            "ref_tokens": ref_prompt.num_visual, "glance_tokens": glance_prompt.num_visual, "glance_grid": (rows, cols),
            "ref_letters": ref[letter_ids].softmax(0).cpu(), "glance_letters": glance[letter_ids].softmax(0).cpu(),
            "glance_kl": kl(ref, glance), "views": [v.box for v in views], "pixels": [v.pixels for v in views],
            "tile_kl": torch.tensor(tile_kl), "tile_letters": torch.stack(tile_letters),
            "detail": torch.tensor(details), "coverage": torch.tensor(covers),
            "per_token": recorder.per_token().cpu().half(), "seconds": time.perf_counter() - start,
        }
        torch.save(record, target)
        torch.cuda.empty_cache()
        print(f"[{number + 1}/{len(samples)}] {sample['id']} views={len(views)} ref_ok="
              f"{int(record['ref_letters'].argmax()) == record['correct']} glance_ok="
              f"{int(record['glance_letters'].argmax()) == record['correct']} "
              f"best_tile_ok={int(record['tile_letters'][:, record['correct']].argmax() >= 0) and bool((record['tile_letters'].argmax(1) == record['correct']).any())} "
              f"{record['seconds']:.1f}s", flush=True)


def extend(target, sample, args, reader, tokenizer, number, total):
    """Coarse views for an existing label record: same glance, same reference, same read-in labels."""
    record = torch.load(target)
    if "scale" in record:
        return
    start = time.perf_counter()
    image = Image.open(sample["image"]).convert("RGB")
    text, correct, letters = question_text(sample)
    letter_ids = torch.tensor([tokenizer.convert_tokens_to_ids(l) for l in letters], device=reader.adapter.device)
    ref, _, _ = reader.log_probs([Piece(resized(image, args.ref_tokens))], text)
    glance_image = resized(image, args.glance_tokens)
    glance_scale = glance_image.width / image.width
    seen = {tuple(p) for p in record["pixels"]}
    scale = [1.0] * len(record["views"])
    views, kls, letters_out, details, covers = [], [], [], [], []
    for factor in args.extend_scales:
        for view in tile_views(image.width, image.height, args.tile_tokens, args.overlap, factor):
            if view.pixels in seen:
                continue
            seen.add(view.pixels)
            crop = resized(view.crop(image), args.tile_tokens)
            x0, y0, x1, y1 = (round(v * 1000) for v in view.box)
            pieces = [Piece(glance_image, "Overview of the whole image:"),
                      Piece(crop, f"Zoomed-in view of region [{x0}, {y0}, {x1}, {y1}] (0-1000 coordinates):")]
            lp, _, _ = reader.log_probs(pieces, text)
            views.append(view)
            scale.append(factor)
            kls.append(kl(ref, lp))
            letters_out.append(lp[letter_ids].softmax(0).cpu())
            details.append(detail_loss(image, view, glance_scale))
            covers.append(coverage(view, sample["bboxes"]))
    if views:
        record["views"] = list(record["views"]) + [v.box for v in views]
        record["pixels"] = list(record["pixels"]) + [v.pixels for v in views]
        record["tile_kl"] = torch.cat([record["tile_kl"], torch.tensor(kls)])
        record["tile_letters"] = torch.cat([record["tile_letters"], torch.stack(letters_out)])
        record["detail"] = torch.cat([record["detail"], torch.tensor(details)])
        record["coverage"] = torch.cat([record["coverage"], torch.tensor(covers)])
    record["scale"] = torch.tensor(scale)
    record["seconds_extend"] = time.perf_counter() - start
    tmp = target.with_suffix(".tmp")
    torch.save(record, tmp)
    tmp.replace(target)
    torch.cuda.empty_cache()
    print(f"[{number + 1}/{total}] {sample['id']} +{len(views)} coarse views {record['seconds_extend']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

"""HiDe (Liu et al., ICML'26, arXiv 2510.00054) re-run on our benchmarks with Qwen2.5-VL, same questions and scoring.

The official Qwen2.5 code in external/HiDe/Hide/Qwen2.5 is imported unmodified:
  modeling_qwen2_5_vl_re_infer  HF Qwen2.5-VL with attention weights computed only for the text query rows at layer
                                SELECT_LAYER=[15] (1-based count of decoder layers, head mean, kept on the CPU)
  Get_box                       get_inputs / messages2out / messages2att / from_img_and_att_get_cropbox
  utiles.process                per-token maps, Gaussian sigma, subtraction of the first 8 "search" tokens' maps
The per-sample driver follows inference.once_infer step by step (that function also runs a direct-answer pass that is
not part of the HiDe answer, and its no-entity branch calls from_img_and_att_get_cropbox with a wrong signature, so the
steps are re-sequenced here instead of calling it):
  1. entity extraction: text-only greedy generation with HiDe's extraction prompt (read verbatim from inference.py)
  2. one prefill on [image, "Search the following entities in the images: <entities>"], attention at the select layer
  3. boxes: per entity token, sigma=3 smoothing, threshold 0.7, connected components; union of all boxes, compacted
     with layout preserved (Layout-Preserving Decoupling) into one image
  4. answer prompt [original image, compact image, question + instruction]; HiDe greedily generates the answer, here
     the answer is the argmax of the option-letter probabilities at the first answer token of that same prompt.

Rows: runs/fovea/<dataset>/hide<tag>.jsonl, one per sample in loader order.

usage: baseline_hide.py --dataset vstar [--model models/Qwen2.5-VL-3B-Instruct] [--limit N] [--tag _3b]
"""
import argparse
import ast
import functools
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # Get_box imports pyplot

import torch
import transformers.integrations.sdpa_attention as _sdpa

# enable_gqa=True has no memory-efficient kernel (and flash is unavailable on this Windows build), so SDPA falls back
# to the math kernel and materialises heads x T x T fp32 scores (16 GB at 16k visual tokens). Repeating K/V first
# (transformers' own non-GQA path) runs the memory-efficient kernel; outputs are the same attention.
_sdpa.use_gqa_in_sdpa = lambda attention_mask, key: False
import torch.nn.functional as F
from PIL import Image

for _stream in (sys.stdout, sys.stderr):  # transformers prints emoji warnings; the Windows console is cp1252
    _stream.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fovea_label import INSTRUCTION, LOADERS, ROOT, question_text  # noqa: E402
from fovea_open import INSTRUCTION as OPEN_INSTRUCTION, match, open_samples  # noqa: E402

HIDE_DIR = ROOT / "external/HiDe/Hide/Qwen2.5"
sys.path.insert(0, str(HIDE_DIR))
import modeling_qwen2_5_vl_re_infer as hide_modeling  # noqa: E402
from Get_box import from_img_and_att_get_cropbox, get_inputs, messages2att, messages2out  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
SUFFIX = "\nAnswer with the option's letter from the given choices directly."  # inference.py, appended to the question
SEARCH = "Search the following entities in the images: "  # inference.py


def extraction_prompt() -> str:
    """The entity-extraction prompt is a local string inside inference.once_infer; read it from the source verbatim."""
    tree = ast.parse((HIDE_DIR / "inference.py").read_text(encoding="utf-8"))
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "once_infer")
    node = next(n for n in func.body if isinstance(n, ast.Assign) and n.targets[0].id == "prompt_ques")
    return node.value.value


class SkipDebugImages:
    """compact_and_center_with_relative_pos writes three full-resolution debug PNGs (case_SPAR_*) into the working
    directory per sample. They are never read back; skip them so they neither litter the repo nor inflate the timing."""

    def __enter__(self):
        self.save = Image.Image.save
        original = self.save

        def save(image, fp, *args, **kwargs):
            if isinstance(fp, (str, Path)) and os.path.basename(str(fp)).startswith("case_SPAR_"):
                return None
            return original(image, fp, *args, **kwargs)

        Image.Image.save = save
        return self

    def __exit__(self, *exc):
        Image.Image.save = self.save


def last_logits_only(model):
    """messages2att calls model(**inputs, output_attentions=True, ...) and ignores the logits; with the default
    logits_to_keep=0 that materialises seq_len x 151k logits (~5 GB at 16k visual tokens). Keep the last position only:
    no effect on any value HiDe uses."""
    forward = model.forward

    @functools.wraps(forward)  # generate() inspects forward's signature (position_ids)
    def wrapped(*args, **kwargs):
        kwargs.setdefault("logits_to_keep", 1)
        return forward(*args, **kwargs)

    model.forward = wrapped


def hide_sample(model, processor, image_path: str, ques: str, prompt_ques: str, sigma: float, thre: float):
    timings = {}
    t0 = time.perf_counter()
    prompt_messages = [{"role": "user", "content": [{"type": "text", "text": prompt_ques.format(input_text=ques)}]}]
    _, _, _, inputs, _ = get_inputs(prompt_messages, processor, model)
    prompt_output_text, _ = messages2out(model, processor, inputs)
    answer_out = prompt_output_text[0].split("<FINAL_OUTPUT>")[-1].split("</FINAL_OUTPUT>")[0]
    timings["extract"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    search = SEARCH + answer_out if answer_out else SEARCH + ques + SUFFIX  # the two branches of once_infer
    messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": search}]}]
    _, _, _, inputs, _ = get_inputs(messages, processor, model)
    attn_visual = int((inputs["input_ids"] == model.config.image_token_id).sum())
    attention, dicts, img_start, img_end = messages2att(model, processor, inputs)
    timings["attention"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with SkipDebugImages():
        results = from_img_and_att_get_cropbox(inputs, attention, dicts, [image_path], img_start, img_end, [sigma], [thre])
    _, _, _, highlight_imgs, bounding_boxes = results[str(sigma)][str(thre)]
    del attention
    timings["boxes"] = time.perf_counter() - t0
    return answer_out, highlight_imgs, bounding_boxes, attn_visual, timings


def sync(model):
    if model.device.type == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="vstar", choices=sorted(LOADERS))
    parser.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--layer", type=int, default=0,
                        help="override SELECT_LAYER (1-based; official code: 15, set for the 7B; 0 keeps it)")
    parser.add_argument("--sigma", type=float, default=3, help="official default 3")
    parser.add_argument("--thre", type=float, default=0.7, help="official default 0.7")
    parser.add_argument("--max-tokens", type=int, default=16384, help="max visual tokens per image (official 16384)")
    parser.add_argument("--min-tokens", type=int, default=256, help="min visual tokens per image (official 256)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="", help="e.g. auto: offload layers to the CPU (7B on a 16 GB card)")
    parser.add_argument("--gpu-mem", type=int, default=10, help="GiB of weights kept on the GPU with --device-map")
    parser.add_argument("--open", action="store_true",
                        help="open-ended subset (fovea_open.py): no options, the answer is generated and matched")
    args = parser.parse_args()
    sigma = int(args.sigma) if float(args.sigma).is_integer() else args.sigma  # results are keyed by str(sigma)

    if args.layer:
        hide_modeling.SELECT_LAYER = [args.layer]
    loader, data_root = LOADERS[args.dataset]
    samples = open_samples(args.dataset) if args.open else loader(ROOT / data_root)
    if args.limit:
        samples = samples[:args.limit]
    out = ROOT / f"runs/fovea/{args.dataset}/hide{'_open' if args.open else ''}{args.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    model_path = str(ROOT / args.model)
    model = hide_modeling.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map=args.device_map or args.device,
        max_memory={0: f"{args.gpu_mem}GiB", "cpu": "48GiB"} if args.device_map else None).eval()
    last_logits_only(model)
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True, min_pixels=args.min_tokens * 28 * 28,
                                              max_pixels=args.max_tokens * 28 * 28)
    tokenizer = processor.tokenizer
    prompt_ques = extraction_prompt()
    print(f"HiDe: layer {hide_modeling.SELECT_LAYER} of {model.config.text_config.num_hidden_layers}, sigma {sigma}, "
          f"threshold {args.thre}, visual tokens {args.min_tokens}-{args.max_tokens}, {len(samples)} samples -> {out}",
          flush=True)

    workdir = tempfile.mkdtemp(prefix="hide_")  # HiDe code writes relative to the working directory
    os.chdir(workdir)
    hits = 0
    with out.open("w", encoding="utf-8") as f:
        for number, sample in enumerate(samples):
            if args.open:
                ques, suffix = sample["question"], "\n" + OPEN_INSTRUCTION
            else:
                text, correct, letters = question_text(sample)
                assert text.endswith("\n" + INSTRUCTION) and SUFFIX == "\n" + INSTRUCTION
                ques, suffix = text[:-len(SUFFIX)], SUFFIX
                letter_ids = torch.tensor([tokenizer.convert_tokens_to_ids(l) for l in letters])
            image_path = str(Path(sample["image"]).resolve())

            sync(model)
            start = time.perf_counter()
            entities, highlight_imgs, boxes, attn_visual, timings = hide_sample(
                model, processor, image_path, ques, prompt_ques, sigma, args.thre)
            t0 = time.perf_counter()
            content = [{"type": "image", "image": image_path}]
            content += [{"type": "image", "image": h} for h in highlight_imgs]
            content.append({"type": "text", "text": ques + suffix})
            _, _, _, inputs, _ = get_inputs([{"role": "user", "content": content}], processor, model)
            if args.open:
                # HiDe's own answer step: greedy generation, matched to the correct option's text
                with torch.no_grad():
                    gen = model.generate(**inputs, max_new_tokens=12, do_sample=False, use_cache=True)
                sync(model)
                timings["answer"] = time.perf_counter() - t0
                answer_text = tokenizer.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                ok = match(answer_text, sample["gold"], sample["others"])
                hits += ok
                visual = int((inputs["input_ids"] == model.config.image_token_id).sum())
                row = {"id": sample["id"], "category": sample.get("category"), "correct": bool(ok),
                       "answer_text": answer_text, "gold": sample["gold"], "seconds": time.perf_counter() - start,
                       "visual_tokens": visual, "visual_tokens_attention": attn_visual,
                       "n_images": 1 + len(highlight_imgs), "entities": entities,
                       "boxes": {str(k): v for k, v in boxes.items()},
                       "timings": {k: round(v, 3) for k, v in timings.items()}}
                f.write(json.dumps(row) + "\n")
                f.flush()
                del inputs, gen
                if model.device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"[{number + 1}/{len(samples)}] {sample['id']} {answer_text!r} gold {sample['gold']!r} "
                      f"{'ok' if ok else 'x'} acc {hits / (number + 1):.3f} {row['seconds']:.1f}s", flush=True)
                continue
            with torch.no_grad():
                logits = model(**inputs, logits_to_keep=1, use_cache=False).logits[0, -1].float()
            sync(model)
            timings["answer"] = time.perf_counter() - t0
            seconds = time.perf_counter() - start

            log_probs = F.log_softmax(logits[letter_ids.to(logits.device)], -1)
            probs = log_probs.exp().cpu().tolist()
            pred = int(log_probs.argmax())
            ok = letters[pred] == correct
            hits += ok
            visual = int((inputs["input_ids"] == model.config.image_token_id).sum())
            row = {"id": sample["id"], "category": sample["category"], "correct": bool(ok),
                   "answer": letters.index(correct), "pred": pred, "probs": probs, "seconds": seconds,
                   "visual_tokens": visual, "visual_tokens_attention": attn_visual, "n_images": 1 + len(highlight_imgs),
                   "top_token": tokenizer.decode(int(logits.argmax())), "entities": entities,
                   "boxes": {str(k): v for k, v in boxes.items()},
                   "timings": {k: round(v, 3) for k, v in timings.items()}}
            f.write(json.dumps(row) + "\n")
            f.flush()
            del inputs, logits
            if model.device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[{number + 1}/{len(samples)}] {sample['id']} ok={ok} acc={hits / (number + 1):.3f} "
                  f"crops={len(highlight_imgs)} visual={visual} {seconds:.1f}s {row['timings']}", flush=True)


if __name__ == "__main__":
    main()

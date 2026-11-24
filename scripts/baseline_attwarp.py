"""AttWarp (Dalal et al., ICLR'26, arXiv 2510.09741) re-run on our benchmarks with Qwen2.5-VL, same questions and scoring.

The released code (external/Attwarp/"Attention Guided Warping") is LLaVA-1.5 only; its warp and mask code is imported
unmodified and only the attention read-out is ported to Qwen2.5-VL:
  attention   attention_extraction/llava.py MaskHookLogger: at every forward of decoder layer L during greedy generation
              (max_new_tokens=20, temperature 0) take the attention row of the last query token, slice the image tokens,
              renormalise the slice per head, mean over heads; the map is the mean over generation steps. Ported as a
              forward hook on the Qwen2.5-VL attention module that recomputes softmax(q_last K^T / sqrt(d)) from the
              layer's own projections, M-RoPE and KV cache (the weights LLaVA's eager attention would return), so the
              model itself runs with SDPA. L = 16 (the paper's Qwen setting; 0-based like the LLaVA code's layers[20]).
  mask        llava.revise_mask (min-max, z-score x enhance_coe=10, sigmoid, 3x3 box filter) on the (h/28, w/28) token
              grid instead of LLaVA's 24x24, llava.toImg -> uint8, llava.invtrans to the original image size (LANCZOS),
              as blend_mask + main.py do.
  warp        new_method.warp_image_by_attention with transform "identity" (main.py / main_batched.py): row and column
              marginals of the mask, CDFs, inverse map, cv2.remap bilinear.
  answer      warped image alone + question (as evaluate_accuracy.py), first-token option-letter probabilities.
Choices the release does not fix for Qwen2.5-VL on high-resolution images (see the report): the warp output keeps the
original image size (the paper's H x W formulation; main.py's 500 x 500 is LLaVA's 336 px input), and both passes use the
Qwen2.5-VL default visual budget (<= 16384 tokens per image) instead of the paper's 512 x 512 Qwen resize.

Rows: runs/fovea/<dataset>/attwarp<tag>.jsonl, one per sample in loader order. probs_orig are the letter probabilities of
the first (unwarped) pass, i.e. the base model with the same prompt.

usage: baseline_attwarp.py --dataset vstar [--model models/Qwen2.5-VL-3B-Instruct] [--limit N] [--tag _3b]
"""
import argparse
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
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
from fovea_label import LOADERS, ROOT, question_text  # noqa: E402
from fovea_open import match, open_samples  # noqa: E402

from qwen_vl_utils import process_vision_info  # noqa: E402
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb, repeat_kv  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
ATTWARP_DIR = ROOT / "external/Attwarp/Attention Guided Warping"


def load_attwarp():
    """new_method.py imports cleanly. attention_extraction/llava.py imports getmask/get_model from functions.py, which
    needs the LLaVA package; those two are not used here, so a stub stands in for functions.py while it loads."""
    sys.path.insert(0, str(ATTWARP_DIR))
    import new_method
    stub = types.ModuleType("functions")
    stub.getmask = stub.get_model = None
    saved = sys.modules.get("functions")
    sys.modules["functions"] = stub
    path = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location("attwarp_llava_util", ATTWARP_DIR / "attention_extraction/llava.py")
        llava_util = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(llava_util)
    finally:
        sys.path[:] = path  # llava.py prepends attention_extraction/ (which holds a llava.py) to sys.path
        if saved is None:
            sys.modules.pop("functions", None)
        else:
            sys.modules["functions"] = saved
    return new_method, llava_util


class LastTokenImageAttention:
    """Port of MaskHookLogger to a Qwen2.5-VL attention module (one image per prompt)."""

    def __init__(self, attn):
        self.attn, self.span, self.steps = attn, None, []
        self.handle = attn.register_forward_hook(self.hook, with_kwargs=True)

    def reset(self, span):
        self.span, self.steps = span, []

    @torch.no_grad()
    def hook(self, module, args, kwargs, output):
        if self.span is None:
            return
        hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        cos, sin = kwargs["position_embeddings"]
        cache = kwargs.get("past_key_values")
        mrope = module.rope_scaling["mrope_section"]
        b = hidden.shape[0]
        q = module.q_proj(hidden[:, -1:]).view(b, 1, -1, module.head_dim).transpose(1, 2)
        q, _ = apply_multimodal_rotary_pos_emb(q, q, cos[..., -1:, :], sin[..., -1:, :], mrope)
        if cache is not None:  # the layer has already appended this step's keys (post-RoPE) to the cache
            keys = cache.layers[module.layer_idx].keys
        else:
            k = module.k_proj(hidden).view(b, hidden.shape[1], -1, module.head_dim).transpose(1, 2)
            keys, _ = apply_multimodal_rotary_pos_emb(k, k, cos, sin, mrope)
        keys = repeat_kv(keys, module.num_key_value_groups)
        weights = torch.softmax((q.float() @ keys.float().transpose(-1, -2)) * module.scaling, dim=-1)  # [b,H,1,T]
        start, end = self.span
        image = weights[0, :, -1, start:end]
        image = image / (image.sum(-1, keepdim=True) + 1e-12)  # renormalise the slice per head, as MaskHookLogger
        self.steps.append(image.mean(0))

    def finalize(self):
        return torch.stack(self.steps).mean(0)


def build_inputs(processor, image, text, max_pixels, device):
    messages = [{"role": "user", "content": [{"type": "image", "image": image, "max_pixels": max_pixels},
                                             {"type": "text", "text": text}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, _ = process_vision_info(messages)
    return processor(text=[prompt], images=images, return_tensors="pt").to(device)


def sync(model):
    if model.device.type == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="vstar", choices=sorted(LOADERS))
    parser.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--layer", type=int, default=16, help="0-based decoder layer (paper, Qwen: 16)")
    parser.add_argument("--enhance-coe", type=float, default=10, help="main.py / llava_api default 10")
    parser.add_argument("--kernel-size", type=int, default=3, help="main.py / llava_api default 3")
    parser.add_argument("--transform", default="identity", help="main.py: identity (paper uses sqrt for GQA only)")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="main.py getmask default 20")
    parser.add_argument("--max-tokens", type=int, default=16384, help="max visual tokens per image (Qwen default)")
    parser.add_argument("--warp-size", default="orig", help="'orig' (H x W of the image) or WxH, e.g. 500x500 (main.py)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="", help="e.g. auto: offload layers to the CPU (7B on a 16 GB card)")
    parser.add_argument("--gpu-mem", type=int, default=10, help="GiB of weights kept on the GPU with --device-map")
    parser.add_argument("--open", action="store_true",
                        help="open-ended subset (fovea_open.py): no options, the answer is generated and matched")
    args = parser.parse_args()

    new_method, llava_util = load_attwarp()
    loader, data_root = LOADERS[args.dataset]
    samples = open_samples(args.dataset) if args.open else loader(ROOT / data_root)
    if args.limit:
        samples = samples[:args.limit]
    out = ROOT / f"runs/fovea/{args.dataset}/attwarp{'_open' if args.open else ''}{args.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    model_path = str(ROOT / args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=args.device_map or args.device,
        max_memory={0: f"{args.gpu_mem}GiB", "cpu": "48GiB"} if args.device_map else None).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    layers = model.model.language_model.layers
    recorder = LastTokenImageAttention(layers[args.layer].self_attn)
    max_pixels = args.max_tokens * 28 * 28
    image_token = model.config.image_token_id
    device = model.device
    print(f"AttWarp: layer {args.layer} (0-based) of {len(layers)}, enhance {args.enhance_coe}, kernel "
          f"{args.kernel_size}, transform {args.transform}, warp {args.warp_size}, <= {args.max_tokens} visual tokens, "
          f"{len(samples)} samples -> {out}", flush=True)

    hits = hits_orig = 0
    with out.open("w", encoding="utf-8") as f:
        for number, sample in enumerate(samples):
            if args.open:
                text = sample["open_text"]
            else:
                text, correct, letters = question_text(sample)
                letter_ids = torch.tensor([tokenizer.convert_tokens_to_ids(l) for l in letters], device=device)
            sync(model)
            start = time.perf_counter()
            image = Image.open(sample["image"]).convert("RGB")

            # pass 1: attention of the generated answer to the image (getmask)
            inputs = build_inputs(processor, image, text, max_pixels, device)
            positions = (inputs["input_ids"][0] == image_token).nonzero().flatten()
            span = (int(positions[0]), int(positions[-1]) + 1)
            assert span[1] - span[0] == len(positions)
            _, grid_h, grid_w = inputs["image_grid_thw"][0].tolist()
            rows, cols = grid_h // 2, grid_w // 2
            recorder.reset(span)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True,
                                     output_logits=True, return_dict_in_generate=True)
            recorder.span = None
            steps = len(recorder.steps)
            att = recorder.finalize().float().cpu().view(rows, cols)
            if not args.open:
                orig_lp = F.log_softmax(gen.logits[0][0, letter_ids].float(), -1)
            answer_text = tokenizer.decode(gen.sequences[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            visual_orig = len(positions)
            del gen, inputs
            t_attention = time.perf_counter() - start

            # mask (blend_mask without the overlay) and warp (save_warped_image without the files)
            t0 = time.perf_counter()
            mask = llava_util.revise_mask(att, kernel_size=args.kernel_size, enhance_coe=args.enhance_coe)
            mask = llava_util.toImg(mask.detach().cpu().reshape(1, rows, cols))
            mask = llava_util.invtrans(mask, image, method=Image.LANCZOS)
            att_map = np.array(mask.convert("L"))
            bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            width, height = image.size if args.warp_size == "orig" else map(int, args.warp_size.split("x"))
            new_method.set_transform_function(args.transform, 1.0, 1.0, False)
            warped = new_method.warp_image_by_attention(bgr, att_map, width, height)
            warped = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            t_warp = time.perf_counter() - t0

            # pass 2: the warped image alone
            t0 = time.perf_counter()
            inputs = build_inputs(processor, warped, text, max_pixels, device)
            if args.open:
                with torch.no_grad():
                    gen = model.generate(**inputs, max_new_tokens=12, do_sample=False, use_cache=True)
                sync(model)
                t_answer = time.perf_counter() - t0
                final = tokenizer.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                ok = match(final, sample["gold"], sample["others"])
                ok_orig = match(answer_text, sample["gold"], sample["others"])
                hits += ok
                hits_orig += ok_orig
                row = {"id": sample["id"], "category": sample.get("category"), "correct": bool(ok),
                       "answer_text": final, "gold": sample["gold"], "pass1_text": answer_text,
                       "correct_orig": bool(ok_orig), "seconds": time.perf_counter() - start,
                       "visual_tokens": int((inputs["input_ids"] == image_token).sum()),
                       "visual_tokens_attention": visual_orig, "attention_steps": steps, "grid": [rows, cols],
                       "timings": {"attention": round(t_attention, 3), "warp": round(t_warp, 3),
                                   "answer": round(t_answer, 3)}}
                f.write(json.dumps(row) + "\n")
                f.flush()
                del inputs, gen
                if model.device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"[{number + 1}/{len(samples)}] {sample['id']} {final!r} gold {sample['gold']!r} ok={ok} "
                      f"acc={hits / (number + 1):.3f} {row['seconds']:.1f}s", flush=True)
                continue
            with torch.no_grad():
                logits = model(**inputs, logits_to_keep=1, use_cache=False).logits[0, -1].float()
            sync(model)
            t_answer = time.perf_counter() - t0
            seconds = time.perf_counter() - start

            log_probs = F.log_softmax(logits[letter_ids], -1)
            pred = int(log_probs.argmax())
            ok = letters[pred] == correct
            ok_orig = letters[int(orig_lp.argmax())] == correct
            hits += ok
            hits_orig += ok_orig
            visual = int((inputs["input_ids"] == image_token).sum())
            row = {"id": sample["id"], "category": sample["category"], "correct": bool(ok),
                   "answer": letters.index(correct), "pred": pred, "probs": log_probs.exp().cpu().tolist(),
                   "seconds": seconds, "visual_tokens": visual, "visual_tokens_attention": visual_orig,
                   "attention_steps": steps, "grid": [rows, cols], "pass1_text": answer_text,
                   "correct_orig": bool(ok_orig), "probs_orig": orig_lp.exp().cpu().tolist(),
                   "top_token": tokenizer.decode(int(logits.argmax())),
                   "timings": {"attention": round(t_attention, 3), "warp": round(t_warp, 3),
                               "answer": round(t_answer, 3)}}
            f.write(json.dumps(row) + "\n")
            f.flush()
            del inputs, logits
            if model.device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[{number + 1}/{len(samples)}] {sample['id']} ok={ok} orig_ok={ok_orig} acc={hits / (number + 1):.3f} "
                  f"orig_acc={hits_orig / (number + 1):.3f} grid={rows}x{cols} steps={steps} {seconds:.1f}s "
                  f"{row['timings']}", flush=True)


if __name__ == "__main__":
    main()

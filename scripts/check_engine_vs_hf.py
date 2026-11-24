"""Checks the layer-by-layer decoder (vzcore.engine) against the Hugging Face forward pass of the same model on the same
inputs: one image already at its final size, one question; compares the last-token logits. Covers the Qwen3-VL port
(q/k norms, interleaved mrope, DeepStack) and guards the Qwen2.5-VL path.

    python scripts/check_engine_vs_hf.py --model models/Qwen2.5-VL-3B-Instruct
    FOVEA_PATCH=32 python scripts/check_engine_vs_hf.py --model models/Qwen3-VL-8B-Instruct
Also checks a fork: the image prefix cached once, then [image, crop, question] prefilled on a copy of that cache.
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from vzcore.models import load_model  # noqa: E402
from fovea.canvas import Piece, resized  # noqa: E402
from fovea_label import LOADERS, Reader, question_text  # noqa: E402
from fovea_cascade import prefill_glance  # noqa: E402


def hf_logits(adapter, pieces, text):
    content = [{"type": "image"} for _ in pieces] + [{"type": "text", "text": text}]
    messages = [{"role": "user", "content": content}]
    prompt = adapter.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                   **getattr(adapter, "chat_kwargs", {}))
    inputs = adapter.processor(text=[prompt], images=[p.image for p in pieces], return_tensors="pt").to(adapter.device)
    out = adapter.model(**inputs, use_cache=False, logits_to_keep=1)
    return out.logits[0, -1].float(), int(inputs["input_ids"].shape[1])


def compare(name, a, b):
    diff = (a - b).abs().max().item()
    top_a, top_b = a.topk(5).indices.tolist(), b.topk(5).indices.tolist()
    pa, pb = a.softmax(-1), b.softmax(-1)
    tv = 0.5 * (pa - pb).abs().sum().item()
    print(f"{name}: max |dlogit| {diff:.3f}  total variation {tv:.4f}  top1 {'same' if top_a[0] == top_b[0] else 'DIFF'}"
          f"  top5 overlap {len(set(top_a) & set(top_b))}/5", flush=True)
    return top_a[0] == top_b[0] and tv < 0.05


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    adapter = load_model("qwen2_5_vl", model_id=str(ROOT / args.model), dtype="bfloat16")
    reader = Reader(adapter)
    ok = True
    for s in LOADERS["vstar"][0](ROOT / LOADERS["vstar"][1])[:args.n]:
        image = Image.open(s["image"]).convert("RGB")
        text, _, _ = question_text(s)
        small = resized(image, 1024)
        lp, prompt, _ = reader.log_probs([Piece(small)], text)
        ref, n = hf_logits(adapter, [Piece(small)], text)
        assert n == prompt.length, (n, prompt.length)
        ok &= compare(f"{s['id']} single image", lp.float(), torch.log_softmax(ref, -1))
        # fork: glance prefix cached, then a crop appended on a copy of the cache
        crop = resized(image.crop((0, 0, image.width // 2, image.height // 2)), 256)
        gprompt, gspans = reader.canvas.build([Piece(small)], text)
        cut = int(gspans[0].max()) + 1
        cache = reader.decoder.new_cache(1, gprompt.length + 1)
        prefill_glance(reader.decoder, gprompt, cut, cache, None)
        fprompt, _ = reader.canvas.build([Piece(small), Piece(crop)], text, separate=True)
        fork = cache.fork(1, fprompt.length - cut + 1)
        fork.length = [cut] * len(fork.length)
        logits = reader.decoder.forward(fprompt.embeds[:, cut:], fprompt.token_index[cut:], fprompt.positions[..., cut:],
                                        fork, None, "prefill")
        ref2, n2 = hf_logits(adapter, [Piece(small), Piece(crop)], text)
        assert n2 == fprompt.length, (n2, fprompt.length)
        ok &= compare(f"{s['id']} fork with a crop", torch.log_softmax(logits[0, -1].float(), -1),
                      torch.log_softmax(ref2, -1))
    print("ENGINE MATCHES HF" if ok else "MISMATCH", flush=True)


if __name__ == "__main__":
    main()

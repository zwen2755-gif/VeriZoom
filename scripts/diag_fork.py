"""Where does a fork differ from the Hugging Face forward? (a) single image, prefill split at the image end (state
continuation of linear-attention layers); (b) two images encoded jointly as Hugging Face does, then the fork.

    FOVEA_PATCH=32 python scripts/diag_fork.py --model models/Qwen3.5-9B
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
from check_engine_vs_hf import compare, hf_logits  # noqa: E402
from fovea.canvas import Piece, resized  # noqa: E402
from fovea_cascade import prefill_glance  # noqa: E402
from fovea_label import LOADERS, Reader, question_text  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-9B")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    adapter = load_model("qwen2_5_vl", model_id=str(ROOT / args.model), dtype="bfloat16")
    reader = Reader(adapter)
    dec = reader.decoder
    for s in LOADERS["vstar"][0](ROOT / LOADERS["vstar"][1])[:args.n]:
        image = Image.open(s["image"]).convert("RGB")
        text, _, _ = question_text(s)
        small = resized(image, 1024)
        # (a) one image, prefill split at the image end
        p, spans = reader.canvas.build([Piece(small)], text)
        cut = int(spans[0].max()) + 1
        cache = dec.new_cache(1, p.length + 1)
        lg = prefill_glance(dec, p, cut, cache, None)
        ref, _ = hf_logits(adapter, [Piece(small)], text)
        compare(f"{s['id']} (a) split prefill, one image", torch.log_softmax(lg[0, -1].float(), -1),
                torch.log_softmax(ref, -1))
        # (b) two images encoded jointly (as Hugging Face), forked after the first image
        crop = resized(image.crop((0, 0, image.width // 2, image.height // 2)), 256)
        reader.canvas.cache.clear()
        fp, fspans = reader.canvas.build([Piece(small), Piece(crop)], text, separate=False)
        cut2 = int(fspans[0].max()) + 1
        cache = dec.new_cache(1, fp.length + 1)
        lg2 = prefill_glance(dec, fp, cut2, cache, None)
        ref2, _ = hf_logits(adapter, [Piece(small), Piece(crop)], text)
        compare(f"{s['id']} (b) joint encoding, split prefill", torch.log_softmax(lg2[0, -1].float(), -1),
                torch.log_softmax(ref2, -1))
        # (d) the same split in Hugging Face itself (its own cache), against our split (b)
        from transformers import DynamicCache
        proc, model = adapter.processor, adapter.model
        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": text}]}]
        prompt = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, **adapter.chat_kwargs)
        inputs = proc(text=[prompt], images=[small, crop], return_tensors="pt").to(adapter.device)
        ids = inputs["input_ids"][0]
        embeds = model.model.get_input_embeddings()(inputs["input_ids"])
        feats = torch.cat(model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"]).pooler_output)
        embeds[0, ids == adapter.image_token_id] = feats.to(embeds.dtype)
        pos, _ = model.model.get_rope_index(inputs["input_ids"], (inputs["input_ids"] == adapter.image_token_id).int(),
                                            inputs["image_grid_thw"], None, inputs["attention_mask"])
        lm = model.model.language_model
        hf_cache = DynamicCache(config=model.config.text_config)
        lm(inputs_embeds=embeds[:, :cut2], position_ids=pos[..., :cut2], past_key_values=hf_cache, use_cache=True)
        out = lm(inputs_embeds=embeds[:, cut2:], position_ids=pos[..., cut2:], past_key_values=hf_cache, use_cache=True)
        hf_split = model.lm_head(out.last_hidden_state[:, -1]).float()[0]
        compare(f"{s['id']} (d) our split vs HF split", torch.log_softmax(lg2[0, -1].float(), -1),
                torch.log_softmax(hf_split, -1))
        # (c) two images encoded separately (our forks), one pass, no split
        reader.canvas.cache_size = 24
        reader.canvas.cache.clear()
        sp, _ = reader.canvas.build([Piece(small), Piece(crop)], text, separate=True)
        cache = dec.new_cache(1, sp.length + 1)
        lg3 = dec.forward(sp.embeds, sp.token_index, sp.positions, cache, None, "prefill")
        compare(f"{s['id']} (c) separate encoding, one pass", torch.log_softmax(lg3[0, -1].float(), -1),
                torch.log_softmax(ref2, -1))


if __name__ == "__main__":
    main()

"""Is a split prefill exact for Qwen3.5 in Hugging Face itself? Prefix [0, cut) with a cache, then the rest, against
one pass, on the same two-image prompt (a crop after the image), as our forks do.

    python scripts/diag_hf_split.py --model models/Qwen3.5-9B
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from transformers import AutoProcessor, DynamicCache, Qwen3_5ForConditionalGeneration  # noqa: E402
from check_engine_vs_hf import compare  # noqa: E402
from fovea.canvas import resized  # noqa: E402
from fovea_label import LOADERS, question_text  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-9B")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    proc = AutoProcessor.from_pretrained(ROOT / args.model)
    proc.image_processor.size = {"shortest_edge": 32 * 32, "longest_edge": 10 ** 9}
    model = Qwen3_5ForConditionalGeneration.from_pretrained(ROOT / args.model, dtype=dtype,
                                                            attn_implementation="sdpa").eval().cuda()
    image_id = proc.tokenizer.convert_tokens_to_ids(proc.image_token)
    for s in LOADERS["vstar"][0](ROOT / LOADERS["vstar"][1])[:3]:
        image = Image.open(s["image"]).convert("RGB")
        text, _, _ = question_text(s)
        small = resized(image, 1024)
        crop = resized(image.crop((0, 0, image.width // 2, image.height // 2)), 256)
        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": text}]}]
        prompt = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        inputs = proc(text=[prompt], images=[small, crop], return_tensors="pt").to("cuda")
        full = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        ids = inputs["input_ids"][0]
        img = (ids == image_id).nonzero().squeeze(1)
        n0 = int(inputs["image_grid_thw"][0].prod()) // 4
        cut = int(img[n0 - 1]) + 1                      # one past the first image's last token
        # embeddings and positions of the whole prompt, then two calls of the language model
        embeds = model.model.get_input_embeddings()(inputs["input_ids"])
        feats = torch.cat(model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"]).pooler_output)
        embeds[0, ids == image_id] = feats.to(embeds.dtype)
        types = (inputs["input_ids"] == image_id).int()
        pos, _ = model.model.get_rope_index(inputs["input_ids"], types, inputs["image_grid_thw"], None,
                                            inputs["attention_mask"])
        lm = model.model.language_model
        cache = DynamicCache(config=model.config.text_config)
        lm(inputs_embeds=embeds[:, :cut], position_ids=pos[..., :cut], past_key_values=cache, use_cache=True)
        out = lm(inputs_embeds=embeds[:, cut:], position_ids=pos[..., cut:], past_key_values=cache, use_cache=True)
        split = model.lm_head(out.last_hidden_state[:, -1]).float()[0]
        one = lm(inputs_embeds=embeds, position_ids=pos, use_cache=False)
        one = model.lm_head(one.last_hidden_state[:, -1]).float()[0]
        compare(f"{s['id']} HF one pass via the LM vs model forward", torch.log_softmax(one, -1),
                torch.log_softmax(full, -1))
        compare(f"{s['id']} HF split prefill vs one pass", torch.log_softmax(split, -1), torch.log_softmax(one, -1))


if __name__ == "__main__":
    main()

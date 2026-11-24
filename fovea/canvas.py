"""A canvas is the multi-resolution input of the sampler: a list of pieces (a low-resolution overview, high-resolution
crops, frames, pages), each an image with an optional caption, encoded by the video LLM's vision tower at its own
resolution and laid out in one prompt. Built on the vzcore engine's Prompt so the same decoder loop runs it."""
import os
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from vzcore.models.base import Prompt

# pixels per LLM token: Qwen2.5-VL 14 px patches 2x2 merged (28); Qwen3-VL 16 px patches 2x2 merged (FOVEA_PATCH=32)
PATCH = int(os.environ.get("FOVEA_PATCH", "28"))


@dataclass
class Piece:
    image: Image.Image | list   # an image, or the frames of a clip (same size, even count), resized to PATCH multiples
    caption: str = ""           # text placed right before the piece (e.g. coordinates of a crop, times of a clip)
    meta: dict = field(default_factory=dict)
    fps: float = 0.0            # clips: true sampling rate of the frames, so temporal positions match seconds

    @property
    def is_video(self) -> bool:
        return isinstance(self.image, list)

    @property
    def first(self) -> Image.Image:
        return self.image[0] if self.is_video else self.image

    @property
    def tokens(self) -> int:
        groups = len(self.image) // 2 if self.is_video else 1
        return groups * (self.first.width // PATCH) * (self.first.height // PATCH)


def fit_size(width: int, height: int, max_tokens: int, min_side: int = PATCH) -> tuple[int, int]:
    """Largest (w, h), multiples of PATCH, keeping the aspect ratio, with at most `max_tokens` LLM tokens and never
    above the native size (rounded)."""
    scale = min(1.0, (max_tokens / ((width / PATCH) * (height / PATCH))) ** 0.5)
    w = max(min_side, int(width * scale / PATCH) * PATCH)
    h = max(min_side, int(height * scale / PATCH) * PATCH)
    while (w // PATCH) * (h // PATCH) > max_tokens:
        if w >= h:
            w -= PATCH
        else:
            h -= PATCH
    return w, h


def resized(image: Image.Image, max_tokens: int) -> Image.Image:
    """`image` at `max_tokens`. Large reductions first box-average by an integer factor while the result stays at
    least twice the target size (Image.reduce, ~5x faster than a direct bicubic on 8K inputs, < 0.3/255 mean pixel
    difference), then bicubic to the exact size. Used by every method alike."""
    w, h = fit_size(image.width, image.height, max_tokens)
    factor = min(image.width // (2 * w), image.height // (2 * h))
    if factor >= 2:
        image = image.reduce(factor)
    return image.resize((w, h), Image.BICUBIC)


class Canvas:
    """Encodes pieces + question with a loaded `vzcore.models.qwen2_5_vl.Qwen25VL` adapter."""

    def __init__(self, adapter):
        self.adapter = adapter
        if getattr(adapter, "patch_px", 28) != PATCH:
            raise RuntimeError(f"{type(adapter).__name__} uses {adapter.patch_px} px per token: run with "
                               f"FOVEA_PATCH={adapter.patch_px} (now {PATCH})")
        processor = adapter.processor
        self.processor = processor
        # pieces arrive at their final size; stop the processor from resizing them again
        image_processor = processor.image_processor
        image_processor.min_pixels, image_processor.max_pixels = PATCH * PATCH, 10 ** 9
        image_processor.size = {"shortest_edge": PATCH * PATCH, "longest_edge": 10 ** 9}
        video_processor = processor.video_processor
        video_processor.min_pixels, video_processor.max_pixels = PATCH * PATCH, 10 ** 9
        video_processor.size = {"shortest_edge": PATCH * PATCH, "longest_edge": 10 ** 9}
        self.image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        self.video_token_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
        self.cache_size = 0   # > 0: keep the vision features of the last N image objects (same PIL object = hit)
        self.cache: OrderedDict = OrderedDict()

    def image_features(self, image: Image.Image):
        """(features (tokens, D), grid (1, 3)) of one image through the processor and the vision tower, cached by
        object identity (the cache holds a reference, so an id cannot be reused while cached)."""
        hit = self.cache.get(id(image))
        if hit is not None and hit[0] is image:
            self.cache.move_to_end(id(image))
            return hit[1], hit[2]
        adapter = self.adapter
        host = "visual" in adapter.offload
        out = self.processor.image_processor(images=[image], return_tensors="pt")
        pixel, grid = out["pixel_values"].to(adapter.device), out["image_grid_thw"].to(adapter.device)
        if hasattr(adapter, "encode_images"):       # Qwen3-VL: features and DeepStack features
            features, deepstack = adapter.encode_images(pixel, grid)[0]
            self.cache[id(image)] = (image, features, grid, deepstack)
        else:
            features = torch.cat(adapter.model.model.get_image_features(pixel.cpu() if host else pixel,
                                                                        grid.cpu() if host else grid))
            self.cache[id(image)] = (image, features, grid)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return features, grid

    def encode_missing(self, images):
        """Encode the images not in the feature cache in one vision-tower call (instead of one call per image) and
        cache each one's features."""
        missing = [im for im in images if not (id(im) in self.cache and self.cache[id(im)][0] is im)]
        if len(missing) < 2:
            return
        adapter = self.adapter
        host = "visual" in adapter.offload
        out = self.processor.image_processor(images=missing, return_tensors="pt")
        pixel, grid = out["pixel_values"].to(adapter.device), out["image_grid_thw"].to(adapter.device)
        if hasattr(adapter, "encode_images"):       # Qwen3-VL: features and DeepStack features
            for image, (f, d), g in zip(missing, adapter.encode_images(pixel, grid), grid):
                self.cache[id(image)] = (image, f, g[None], d)
        else:
            features = adapter.model.model.get_image_features(pixel.cpu() if host else pixel,
                                                              grid.cpu() if host else grid)
            for image, f, g in zip(missing, features, grid):
                self.cache[id(image)] = (image, f, g[None])
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    @torch.no_grad()
    def build(self, pieces: list[Piece], text: str, system: str | None = None,
              separate: bool = False) -> tuple[Prompt, list[torch.Tensor]]:
        """-> prompt, and for every piece the (rows*cols,) prompt indices of its visual tokens in raster order.
        separate: encode every image on its own through the feature cache (reuses an image already encoded, e.g. the
        glance image as the overview of a read), instead of one joint encoding of all images."""
        adapter = self.adapter
        content = []
        for piece in pieces:
            if piece.caption:
                content.append({"type": "text", "text": piece.caption})
            content.append({"type": "video" if piece.is_video else "image"})
        content.append({"type": "text", "text": text})
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                          **getattr(adapter, "chat_kwargs", {}))
        images = [p.image for p in pieces if not p.is_video]
        videos = [np.stack([np.asarray(f) for f in p.image]) for p in pieces if p.is_video]
        extra = {}
        if videos:
            extra = {"videos": videos, "do_sample_frames": False,
                     "fps": [p.fps or 2.0 for p in pieces if p.is_video]}
        visual_model = adapter.model.model
        host = "visual" in adapter.offload
        cached = None
        if self.cache_size and (len(images) == 1 or separate) and not videos:
            # cached path (single-image prompts only, bit-identical): the image is encoded once and reused, e.g. by
            # the question and generic glances. Multi-image prompts keep the joint encoding the labels used: bf16
            # vision features of jointly and separately encoded images differ (mean cosine 0.997, min 0.85).
            if separate:
                self.encode_missing(images)
            cached = [self.image_features(image) for image in images]
            merge = self.processor.image_processor.merge_size ** 2
            pad = self.processor.image_token
            parts = prompt_text.split(pad)
            assert len(parts) == len(cached) + 1
            prompt_text = parts[0] + "".join("<|FOVEA_PAD|>" * (int(grid.prod()) // merge) + rest
                                             for (_, grid), rest in zip(cached, parts[1:]))
            prompt_text = prompt_text.replace("<|FOVEA_PAD|>", pad)
            inputs = self.processor.tokenizer([prompt_text], return_tensors="pt").to(adapter.device)
            image_grid, video_grid = torch.cat([grid for _, grid in cached]), None
        else:
            inputs = self.processor(text=[prompt_text], images=images or None, return_tensors="pt", **extra)
            inputs = inputs.to(adapter.device)
            image_grid, video_grid = inputs.get("image_grid_thw"), inputs.get("video_grid_thw")

        ids = inputs["input_ids"]
        embeds = adapter.embed_tokens(ids)
        image_deepstack = None                      # (layers, image tokens, D), Qwen3-VL only
        if cached is not None:
            embeds[0, ids[0] == self.image_token_id] = torch.cat([f for f, _ in cached]).to(embeds.device, embeds.dtype)
            if getattr(adapter, "deepstack_layers", 0):
                image_deepstack = torch.cat([self.cache[id(image)][3] for image in images], 1)
        elif images and hasattr(adapter, "encode_images"):
            encoded = adapter.encode_images(inputs["pixel_values"], image_grid)
            embeds[0, ids[0] == self.image_token_id] = torch.cat([f for f, _ in encoded]).to(embeds.device,
                                                                                             embeds.dtype)
            if getattr(adapter, "deepstack_layers", 0):
                image_deepstack = torch.cat([d for _, d in encoded], 1)
        elif images:
            pixel = inputs["pixel_values"]
            features = torch.cat(visual_model.get_image_features(pixel.cpu() if host else pixel,
                                                                 image_grid.cpu() if host else image_grid))
            embeds[0, ids[0] == self.image_token_id] = features.to(embeds.device, embeds.dtype)
        if videos:
            pixel = inputs["pixel_values_videos"]
            features = torch.cat(visual_model.get_video_features(pixel.cpu() if host else pixel,
                                                                 video_grid.cpu() if host else video_grid))
            embeds[0, ids[0] == self.video_token_id] = features.to(embeds.device, embeds.dtype)
        visual = (ids[0] == self.image_token_id) | (ids[0] == self.video_token_id)
        if hasattr(adapter, "rope_index"):
            positions, _ = adapter.rope_index(ids, image_grid, video_grid, inputs.get("second_per_grid_ts"),
                                              inputs["attention_mask"])
        else:
            positions, _ = visual_model.get_rope_index(ids, image_grid, video_grid, inputs.get("second_per_grid_ts"),
                                                       inputs["attention_mask"])
        deepstack = None
        if image_deepstack is not None:
            deepstack = torch.zeros(image_deepstack.shape[0], ids.shape[1], image_deepstack.shape[2],
                                    dtype=embeds.dtype, device=embeds.device)
            deepstack[:, ids[0] == self.image_token_id] = image_deepstack.to(embeds.device, embeds.dtype)
            # the decoder adds these to the visual tokens of the next prefill (vzcore.engine.forward)
            adapter.active_deepstack, adapter.active_visual = deepstack, ids[0] == self.image_token_id

        slots = visual.nonzero().squeeze(1)
        if self.faithful and all("box" in p.meta for p in pieces):
            positions = self.faithful_positions(ids[0], slots, pieces, positions)
        spans, frame, cell, start = [], torch.full_like(ids[0], -1), torch.full_like(ids[0], -1), 0
        for index, piece in enumerate(pieces):
            span = slots[start:start + piece.tokens]
            spans.append(span)
            frame[span] = index
            cell[span] = torch.arange(piece.tokens, device=span.device)
            start += piece.tokens
        assert start == len(slots)
        length = ids.shape[1]
        prompt = Prompt(ids[0], embeds, positions, visual, frame, cell, torch.arange(length, device=adapter.device),
                        (len(pieces), 0, 0), int(positions.max()) + 1, length, deepstack=deepstack)
        return prompt, spans

    faithful = False  # coordinate-faithful positions: every piece in the source image's own coordinate frame

    def faithful_positions(self, ids, slots, pieces, positions):
        """All visual tokens share one base position; a token's height/width positions are the centre of the source
        region it shows, in units of the source image's native token grid (pieces carry meta box = normalized region
        and meta grid = (rows, cols) of the native token grid). Text keeps its order and starts after the image."""
        positions = positions.clone().float()
        rows_native, cols_native = pieces[0].meta["grid"]
        first, last = int(slots[0]), int(slots[-1])
        base = float(positions[0, 0, first])
        start, coords = 0, []
        for piece in pieces:
            x0, y0, x1, y1 = piece.meta["box"]
            rows, cols = piece.first.height // PATCH, piece.first.width // PATCH
            r = (torch.arange(rows, device=ids.device).float() + 0.5) / rows
            c = (torch.arange(cols, device=ids.device).float() + 0.5) / cols
            h = (y0 + r * (y1 - y0)) * rows_native - 0.5
            w = (x0 + c * (x1 - x0)) * cols_native - 0.5
            hh, ww = torch.meshgrid(h, w, indexing="ij")
            coords.append(torch.stack([torch.zeros_like(hh.flatten()), hh.flatten(), ww.flatten()]))
            start += rows * cols
        coords = torch.cat(coords, 1)                       # (3, Nv)
        visual_positions = base + coords
        positions[:, 0, slots] = visual_positions
        # text between pieces (captions) keeps the order but sits at the base; text after the image follows its end
        between = torch.arange(first, last + 1, device=ids.device)
        between = between[~torch.isin(between, slots)]
        positions[:, 0, between] = base
        tail = float(visual_positions.max()) + 1
        rest = torch.arange(last + 1, len(ids), device=ids.device)
        positions[:, 0, rest] = tail + torch.arange(len(rest), device=ids.device).float()
        return positions


def to_pil(array_or_image) -> Image.Image:
    if isinstance(array_or_image, Image.Image):
        return array_or_image.convert("RGB")
    return Image.fromarray(np.asarray(array_or_image)).convert("RGB")

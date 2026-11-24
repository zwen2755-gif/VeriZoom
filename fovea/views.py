"""Views of a large input and the cheap signals the sampler reads from one glance prefill.

A view is a region of the input that can be read at high resolution: an image tile here; a frame or a clip for
video; a page for documents. Each view has normalized coordinates, so the sampler can report where it looked.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageChops

from .canvas import PATCH


@dataclass
class View:
    box: tuple[float, float, float, float]  # normalized x0, y0, x1, y1
    pixels: tuple[int, int, int, int]       # x0, y0, x1, y1 in the source image

    def crop(self, image: Image.Image) -> Image.Image:
        return image.crop(self.pixels)


def tile_views(width: int, height: int, tile_tokens: int, overlap: float = 0.0, scale: float = 1.0) -> list[View]:
    """A grid of tiles, each about `tile_tokens` LLM tokens when read at native resolution (square-ish), with the
    given fractional overlap between neighbours. scale > 1 widens the footprint (read downsampled to the same
    tokens): coarse views that keep several objects and their layout together."""
    side = min(int((tile_tokens ** 0.5) * PATCH * scale), max(width, height))
    stride = max(PATCH, int(side * (1 - overlap)))

    def starts(total):
        if total <= side:
            return [0]
        count = int(np.ceil((total - side) / stride)) + 1
        return [round(i * (total - side) / (count - 1)) for i in range(count)]

    views = []
    for y in starts(height):
        for x in starts(width):
            x1, y1 = min(width, x + side), min(height, y + side)
            views.append(View((x / width, y / height, x1 / width, y1 / height), (x, y, x1, y1)))
    return views


def detail_loss(image: Image.Image, view: View, glance_scale: float) -> float:
    """Relative high-frequency content of the view that the glance resolution cannot show: mean absolute
    difference between the view and its glance-resolution version upsampled back (on a gray copy of at most 512 px)."""
    crop = view.crop(image).convert("L")
    if max(crop.size) > 512:
        factor = 512 / max(crop.size)
        crop = crop.resize((max(1, int(crop.width * factor)), max(1, int(crop.height * factor))), Image.BILINEAR)
        scale = glance_scale / factor
    else:
        scale = glance_scale
    small = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.BILINEAR)
    back = small.resize(crop.size, Image.BILINEAR)
    a = np.asarray(crop, dtype=np.float32)
    return float(np.abs(a - np.asarray(back, dtype=np.float32)).mean() / 255.0)


def work_copy(image: Image.Image, work: int = 2048) -> Image.Image:
    """A cheap integer box-downsampled copy of at most about 2x `work` px (Image.reduce), for image statistics."""
    factor = max(1, max(image.size) // work)
    return image.reduce(factor) if factor > 1 else image


def detail_fast(image: Image.Image, views: list[View], glance_scale: float, work: int = 2048,
                block: int = 4) -> torch.Tensor:
    """detail_loss for all views at once: a gray copy box-reduced by an integer factor to at most `work` px, its
    glance-resolution version upsampled back, the uint8 absolute difference summed over `block` x `block` cells and
    integrated once; each view is an O(1) box sum (boundaries quantised to one cell). 9-45 ms per image on V* to 8K
    (v1, a bilinear work copy and a full-resolution float integral: 40-130 ms; within-image correlation >= 0.994)."""
    factor = math.ceil(max(image.size) / work)
    gray = (image.reduce(factor) if factor > 1 else image).convert("L")
    scale = glance_scale * image.width / gray.width
    small = gray.resize((max(1, int(gray.width * scale)), max(1, int(gray.height * scale))), Image.BILINEAR)
    diff = np.asarray(ImageChops.difference(gray, small.resize(gray.size, Image.BILINEAR)))
    h, w = max(block, diff.shape[0] // block * block), max(block, diff.shape[1] // block * block)
    diff = np.pad(diff, ((0, max(0, h - diff.shape[0])), (0, max(0, w - diff.shape[1]))))[:h, :w]
    cells = diff.reshape(h // block, block, w // block, block).sum((1, 3), dtype=np.int64)
    rows, cols = cells.shape
    integral = np.zeros((rows + 1, cols + 1), np.int64)
    integral[1:, 1:] = cells.cumsum(0).cumsum(1)
    out = []
    for view in views:
        x0, y0, x1, y1 = view.box
        c0, r0 = min(cols - 1, int(x0 * cols)), min(rows - 1, int(y0 * rows))
        c1, r1 = max(c0 + 1, int(round(x1 * cols))), max(r0 + 1, int(round(y1 * rows)))
        total = integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0]
        out.append(total / ((r1 - r0) * (c1 - c0) * block * block * 255.0))
    return torch.tensor(out, dtype=torch.float32)


class GlanceDone(Exception):
    """Raised by a GlanceRecorder with `stop_after` once it has recorded that many layers: ends the prefill early."""


class GlanceRecorder:
    """Driven by the decoder as a compressor (never evicts): in every prefill layer, records for the visual tokens of
    the glance piece the attention they receive from the question rows (mean and max over heads) and their key and
    value norms (mean over KV heads)."""

    static_after_prefill = True

    def __init__(self, glance_tokens: torch.Tensor, question_rows: torch.Tensor, stop_after: int | None = None,
                 per_head: bool = False):
        self.glance_tokens, self.question_rows, self.stop_after = glance_tokens, question_rows, stop_after
        self.layers: list[torch.Tensor] = []
        self.heads: list[torch.Tensor] | None = [] if per_head else None  # (H_q, Ng) per layer, when per_head
        self.layer_ids: list[int] = []  # decoder layer of every record (all layers, or the attention layers only)

    def on_layer(self, context):
        if context.phase != "prefill":
            return None
        self.layer_ids.append(context.layer)
        slots = self.glance_tokens  # prefill: key slot == prompt index
        attention = context.probabilities(self.question_rows)[0][..., slots].sum(1)  # (H_q, Ng)
        keys = context.key[0][:, slots].float().norm(dim=-1).mean(0)
        values = context.value[0][:, slots].float().norm(dim=-1).mean(0)
        self.layers.append(torch.stack([attention.mean(0), attention.max(0).values, keys, values]))
        if self.heads is not None:
            self.heads.append(attention.half())
        if self.stop_after is not None and len(self.layers) >= self.stop_after:
            raise GlanceDone
        return None

    def per_token(self) -> torch.Tensor:
        return torch.stack(self.layers)  # (L, 4, Ng)

    def per_head(self) -> torch.Tensor:
        return torch.stack(self.heads)  # (L, H_q, Ng)


def token_owner(views: list[View], rows: int, cols: int) -> torch.Tensor:
    """(V, rows*cols) bool: which glance tokens have their centre inside each view."""
    ys = (torch.arange(rows).float() + 0.5) / rows
    xs = (torch.arange(cols).float() + 0.5) / cols
    cy, cx = torch.meshgrid(ys, xs, indexing="ij")
    cy, cx = cy.flatten(), cx.flatten()
    masks = []
    for view in views:
        x0, y0, x1, y1 = view.box
        mask = (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
        if not mask.any():  # a view thinner than one glance token: take the nearest token
            mask[((cx - (x0 + x1) / 2) ** 2 + (cy - (y0 + y1) / 2) ** 2).argmin()] = True
        masks.append(mask)
    return torch.stack(masks)


def view_features(per_token: torch.Tensor, owner: torch.Tensor) -> torch.Tensor:
    """per_token (L, 4, Ng), owner (V, Ng) -> (V, L, 4): attention as log share of the view's summed attention
    relative to the mean view, norms as the view's mean z-scored across views."""
    owner = owner.to(per_token)
    counts = owner.sum(1).clamp_min(1)
    pooled = torch.einsum("lfn,vn->vlf", per_token, owner)
    attention = pooled[..., :2]
    attention = torch.log((attention + 1e-9) / (attention.mean(0, keepdim=True) + 1e-9))
    norms = pooled[..., 2:] / counts[:, None, None]
    norms = (norms - norms.mean(0, keepdim=True)) / (norms.std(0, keepdim=True) + 1e-6)
    return torch.cat([attention, norms], -1)


def depth_bins(features: torch.Tensor, bins: int) -> torch.Tensor:
    """(V, L, F) -> (V, bins, F): average over spans of relative depth (layer l of L goes to floor(bins(l+0.5)/L))."""
    layers = features.shape[1]
    owner = torch.floor(bins * (torch.arange(layers, dtype=torch.float32) + 0.5) / layers).long().clamp_max(bins - 1)
    pooled = features.new_zeros(features.shape[0], bins, features.shape[2]).index_add_(1, owner.to(features.device),
                                                                                         features)
    return pooled / torch.bincount(owner, minlength=bins).to(features).view(1, bins, 1)

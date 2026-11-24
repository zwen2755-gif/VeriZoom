"""The decoder loop. It runs the HF decoder layers' weights through our own attention over `KVCache`, and calls the
compressor between attention and the residual update of every layer. This is the only place compression touches
the model, so every method sees the same forward pass."""
from dataclasses import dataclass

import torch

from . import attention as attn
from .cache import KVCache


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], -1)


@dataclass
class LayerContext:
    """What a compressor sees at one layer: the layer input, post-rope queries, and the keys attention just read."""
    layer: int
    phase: str                  # "prefill", "decode", or "replay" (intervention analysis; no compression)
    step: int                   # decoding step, 0 during prefill
    hidden: torch.Tensor        # (B, Q, D) layer input
    query: torch.Tensor         # (B, H_q, Q, D)
    key: torch.Tensor           # (B, H_kv, N, D) cache view read by attention
    value: torch.Tensor
    query_index: torch.Tensor   # (Q,) original token index of each query
    key_index: torch.Tensor     # (N,)
    key_mask: torch.Tensor | None
    cache: KVCache

    def probabilities(self, rows: slice | torch.Tensor = slice(None)) -> torch.Tensor:
        """(B, H_q, Q', N) float32 attention of the selected query rows over the keys of the view."""
        return attn.probabilities(self.query[:, :, rows], self.key, self.query_index[rows], self.key_index,
                                  self.key_mask)

    def received(self, rows: torch.Tensor | None = None, weights: torch.Tensor | None = None) -> torch.Tensor:
        """(B, H_q, N) attention each key received from the queries of this call (or the given rows, each counted
        `weights` times)."""
        if rows is None:
            return attn.received(self.query, self.key, self.query_index, self.key_index, self.key_mask, weights)
        return attn.received(self.query[:, :, rows], self.key, self.query_index[rows], self.key_index,
                             self.key_mask, weights)


class Decoder:
    def __init__(self, adapter):
        self.adapter = adapter
        self.spec = adapter.spec

    def new_cache(self, batch: int, capacity: int) -> KVCache:
        spec, weight = self.spec, self.adapter.lm_head.weight
        return KVCache(spec.num_layers, batch, spec.num_kv_heads, spec.head_dim, capacity, weight.dtype,
                       weight.device)

    @torch.no_grad()
    def forward(self, embeds: torch.Tensor, query_index: torch.Tensor, positions: torch.Tensor, cache: KVCache,
                compressor=None, phase: str = "prefill", step: int = 0, all_logits: bool = False,
                accountant=None) -> torch.Tensor:
        """embeds (B, Q, D) at original indices `query_index` and rope `positions` (P, B, Q) -> float32 logits
        (B, Q or 1, V). The compressor may return a subset of query rows to carry into the next layer."""
        adapter, spec = self.adapter, self.spec
        hidden = embeds
        deepstack = getattr(adapter, "active_deepstack", None) if phase == "prefill" else None
        if deepstack is not None:
            # rows of this chunk that are visual tokens of the prompt the canvas built last (indices are original
            # token indices; generated or foreign tokens fall outside and get nothing)
            inside = query_index < deepstack.shape[1]
            ds_rows = inside.clone()
            ds_rows[inside] = adapter.active_visual[query_index[inside]]
            if not bool(ds_rows.any()):
                deepstack = None
        cos, sin = adapter.rope(positions, hidden)
        cos, sin = cos[:, None], sin[:, None]
        rotary = cos.shape[-1]
        for index, layer in enumerate(adapter.layers):
            batch, queries, _ = hidden.shape
            if hasattr(layer, "linear_attn"):
                # Qwen3.5 Gated DeltaNet: the model's own module on the recurrent states kept in `cache`; it has no
                # attention to record, so compressors are not called here
                hidden = hidden + layer.linear_attn(hidden_states=layer.input_layernorm(hidden), cache=cache)
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
                continue
            module = layer.self_attn
            x = layer.input_layernorm(hidden)
            q = module.q_proj(x)
            gate = None
            if q.shape[-1] == 2 * spec.num_heads * spec.head_dim:
                # Qwen3.5 gated attention: each head's query is followed by its output gate
                q, gate = torch.chunk(q.view(batch, queries, spec.num_heads, 2 * spec.head_dim), 2, dim=-1)
                gate = gate.reshape(batch, queries, -1)
                q = q.transpose(1, 2)
            else:
                q = q.view(batch, queries, spec.num_heads, spec.head_dim).transpose(1, 2)
            k = module.k_proj(x).view(batch, queries, spec.num_kv_heads, spec.head_dim).transpose(1, 2)
            v = module.v_proj(x).view(batch, queries, spec.num_kv_heads, spec.head_dim).transpose(1, 2)
            if hasattr(module, "q_norm"):  # Qwen3-VL, Qwen3.5: RMS norm over each head's dims before rope
                q, k = module.q_norm(q), module.k_norm(k)
            rot = getattr(adapter, "rotate", rotate_half)   # GLM-4V rotates interleaved pairs of dims
            if rotary == spec.head_dim:
                q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
            else:  # partial rotary (Qwen3.5, GLM-4V): only the first `rotary` dims of each head are rotated
                q = torch.cat([q[..., :rotary] * cos + rot(q[..., :rotary]) * sin, q[..., rotary:]], -1)
                k = torch.cat([k[..., :rotary] * cos + rot(k[..., :rotary]) * sin, k[..., rotary:]], -1)

            square = cache.length[index] == 0
            cache.append(index, k, v, query_index)
            key, value, key_index, key_mask = cache.view(index)
            out = attn.attend(q, key, value, query_index, key_index, key_mask, square)
            if accountant is not None:
                accountant.record(index, phase, batch, queries, key_index, key_mask)
            keep = None
            if compressor is not None:
                context = LayerContext(index, phase, step, hidden, q, key, value, query_index, key_index, key_mask,
                                       cache)
                keep = compressor.on_layer(context)

            out = out.transpose(1, 2).reshape(batch, queries, -1)
            if gate is not None:
                out = out * torch.sigmoid(gate)
            out = module.o_proj(out)
            if hasattr(layer, "post_self_attn_layernorm"):   # GLM-4V: sandwich norms on both residual branches
                out = layer.post_self_attn_layernorm(out)
            hidden = hidden + out
            mlp = layer.mlp(layer.post_attention_layernorm(hidden))
            if hasattr(layer, "post_mlp_layernorm"):
                mlp = layer.post_mlp_layernorm(mlp)
            hidden = hidden + mlp
            if deepstack is not None and index < deepstack.shape[0]:
                # Qwen3-VL DeepStack: add the intermediate vision features to the visual tokens of this chunk
                hidden[:, ds_rows] = hidden[:, ds_rows] + deepstack[index, query_index[ds_rows]].to(hidden.dtype)
            if keep is not None:  # token pruning inside the prefill (FastV): later layers only see `keep`
                hidden, query_index = hidden[:, keep], query_index[keep]
                cos, sin = cos[:, :, keep], sin[:, :, keep]
        if not all_logits:
            hidden = hidden[:, -1:]
        return adapter.lm_head(adapter.norm(hidden)).float()

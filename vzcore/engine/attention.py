"""Attention over the KV cache view, and explicit attention probabilities for the rows a compressor asks about.

The main path uses fused SDPA and never materializes attention maps; compressors that score tokens by attention
call `probabilities` / `received` for the few layers and query rows they need.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right


def visibility(query_index: torch.Tensor, key_index: torch.Tensor, key_mask: torch.Tensor | None,
               groups: int) -> torch.Tensor:
    """Bool mask (B or 1, H_q or 1, Q, N): causal by original token index, AND the cache's key mask."""
    mask = (key_index[None, :] <= query_index[:, None])[None, None]
    if key_mask is not None:
        mask = mask & key_mask.repeat_interleave(groups, 1)[:, :, None, :]
    return mask


REPEAT_KV_PREFILL = False  # set by fovea: SDPA with enable_gqa falls back to the math kernel (Q x N maps in memory)


# opt-in (fovea --reuse-glance): a prefill that continues a cache whose keys are tokens 0..N-1, with the queries the
# last Q of them, runs a bottom-right-aligned causal kernel instead of an explicit (Q, N) mask (8x faster at Q=2.5k)
LOWER_RIGHT_CAUSAL = False


def attend(q, k, v, query_index, key_index, key_mask=None, square: bool = False) -> torch.Tensor:
    """q (B, H_q, Q, D), k/v (B, H_kv, N, D) -> (B, H_q, Q, D).

    `square` means queries and keys are the same tokens in the same order (prefill), so a plain causal kernel
    applies. A single query with no key mask sees every key, as in ordinary decoding.
    """
    if REPEAT_KV_PREFILL and key_mask is None and square:
        groups = q.shape[1] // k.shape[1]
        return F.scaled_dot_product_attention(q, k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1),
                                              is_causal=True)
    if key_mask is None and (square or q.shape[2] == 1):
        return F.scaled_dot_product_attention(q, k, v, is_causal=square, enable_gqa=True)
    if LOWER_RIGHT_CAUSAL and key_mask is None and q.shape[2] > 1:
        n, m = k.shape[2], q.shape[2]
        ends = torch.stack([key_index[0], key_index[-1], query_index[0], query_index[-1]]).tolist()
        if key_index.shape[0] == n and ends == [0, n - 1, n - m, n - 1]:
            groups = q.shape[1] // k.shape[1]
            return F.scaled_dot_product_attention(q, k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1),
                                                  attn_mask=causal_lower_right(m, n))
    mask = visibility(query_index, key_index, key_mask, q.shape[1] // k.shape[1])
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)


def probabilities(q, k, query_index, key_index, key_mask=None) -> torch.Tensor:
    """Softmax attention (B, H_q, Q, N) in float32 for the given query rows."""
    groups = q.shape[1] // k.shape[1]
    scores = q.float() @ k.float().repeat_interleave(groups, 1).transpose(-1, -2) * q.shape[-1] ** -0.5
    scores = scores.masked_fill(~visibility(query_index, key_index, key_mask, groups), float("-inf"))
    return scores.softmax(-1)


def received(q, k, query_index, key_index, key_mask=None, weights: torch.Tensor | None = None,
             chunk: int = 512) -> torch.Tensor:
    """Attention each key receives, summed over the query rows (weighted by `weights` (Q,) if given):
    (B, H_q, N) float32, computed in row chunks. Same arithmetic as `probabilities`, with the expanded keys built
    once instead of once per chunk."""
    groups = q.shape[1] // k.shape[1]
    keys = k.float().repeat_interleave(groups, 1).transpose(-1, -2)
    total = None
    for start in range(0, q.shape[2], chunk):
        rows = slice(start, start + chunk)
        scores = q[:, :, rows].float() @ keys * q.shape[-1] ** -0.5
        scores = scores.masked_fill(~visibility(query_index[rows], key_index, key_mask, groups), float("-inf"))
        scores = scores.softmax(-1)
        part = scores.sum(2) if weights is None else (scores * weights[rows, None]).sum(2)
        total = part if total is None else total + part
    return total

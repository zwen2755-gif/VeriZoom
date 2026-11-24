"""KV cache with per-layer token bookkeeping, so compressors can evict, mask, or temporarily deselect entries.

Each layer stores K/V in a preallocated buffer (B, H_kv, capacity, D) together with the original token index of
every slot. Three ways to drop entries, from permanent to temporary:

- `retain` / `evict`: physical compaction of a layer (one-shot eviction, FastV, H2O). Memory is actually freed
  from the attended range.
- `gather_heads`: per-KV-head compaction when every head keeps the same number of entries (SnapKV, PyramidKV).
  Slot j then holds a different token in each head; the layer's token index is head 0's, which is exact for the
  two things it is still used for after the prefill: causality (every cached entry precedes later queries) and
  counting (every head holds the same number of visual and text entries).
- `head_mask`: per-KV-head eviction with a different number of entries per head (Ada-KV). Entries stay in the
  buffer but are masked out of attention; accounting and `nbytes` count them as evicted, as a variable-length
  cache would store them.
- `select`: a view for the current decoding step only (DyCoke's dynamic pruning); deselected entries stay
  restorable at later steps.

`row_mask` is keyed by token index and masks different tokens in different batch rows, which is how intervention
analysis simulates many evictions of one prefix at once.
"""
import torch


class KVCache:
    def __init__(self, num_layers: int, batch: int, kv_heads: int, head_dim: int, capacity: int,
                 dtype: torch.dtype, device):
        shape = (batch, kv_heads, capacity, head_dim)
        self.k = [torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self.v = [torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self.index = [torch.empty(capacity, dtype=torch.long, device=device) for _ in range(num_layers)]
        self.length = [0] * num_layers
        self.head_mask: list[torch.Tensor | None] = [None] * num_layers  # (H_kv, capacity) bool, True = kept
        self.selected: list[tuple[torch.Tensor, int] | None] = [None] * num_layers
        self.row_mask: torch.Tensor | None = None  # (B, token_count) bool, True = visible
        self.batch, self.kv_heads = batch, kv_heads

    @property
    def num_layers(self) -> int:
        return len(self.k)

    def _grow(self, layer: int, needed: int):
        capacity = self.k[layer].shape[2]
        if needed <= capacity:
            return
        extra = max(needed - capacity, capacity // 2)
        pad = lambda t: torch.cat([t, t.new_empty(*t.shape[:2], extra, t.shape[3])], 2)  # noqa: E731
        self.k[layer], self.v[layer] = pad(self.k[layer]), pad(self.v[layer])
        self.index[layer] = torch.cat([self.index[layer], self.index[layer].new_empty(extra)])
        if self.head_mask[layer] is not None:
            self.head_mask[layer] = torch.cat([self.head_mask[layer],
                                               self.head_mask[layer].new_ones(self.kv_heads, extra)], 1)

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor, token_index: torch.Tensor):
        start, count = self.length[layer], k.shape[2]
        self._grow(layer, start + count)
        self.k[layer][:, :, start:start + count] = k
        self.v[layer][:, :, start:start + count] = v
        self.index[layer][start:start + count] = token_index
        if self.head_mask[layer] is not None:
            self.head_mask[layer][:, start:start + count] = True
        self.length[layer] = start + count

    def tokens(self, layer: int) -> torch.Tensor:
        """Original token index of every stored slot of a layer."""
        return self.index[layer][:self.length[layer]]

    def view(self, layer: int):
        """K, V, token index, and an optional key mask (B, H_kv, N) of the entries attention should read."""
        n = self.length[layer]
        k, v, index = self.k[layer][:, :, :n], self.v[layer][:, :, :n], self.index[layer][:n]
        head_mask = None if self.head_mask[layer] is None else self.head_mask[layer][:, :n]
        if self.selected[layer] is not None:
            slots, upto = self.selected[layer]
            slots = torch.cat([slots, torch.arange(upto, n, device=slots.device)])
            k, v, index = k[:, :, slots], v[:, :, slots], index[slots]
            head_mask = None if head_mask is None else head_mask[:, slots]
        mask = None
        if head_mask is not None:
            mask = head_mask[None].expand(self.batch, -1, -1)
        if self.row_mask is not None:
            rows = self.row_mask[:, index][:, None].expand(-1, self.kv_heads, -1)
            mask = rows if mask is None else mask & rows
        return k, v, index, mask

    # ---- recurrent state of linear-attention layers (Qwen3.5 Gated DeltaNet). Same contract as the Hugging Face
    # linear-attention cache layer, so the model's own GatedDeltaNet module runs on this cache unchanged. A recurrent
    # state cannot be truncated like K/V entries, so a prefix is snapshotted (`snapshot_linear`) and forks start from it.
    def _linear(self) -> dict:
        if not hasattr(self, "linear"):
            self.linear, self.linear_snapshot = {}, None
        return self.linear

    def get_conv_state(self, layer_idx: int, state_idx: int = 0):
        return self._linear().get(layer_idx, {}).get("conv")

    def get_recurrent_state(self, layer_idx: int, state_idx: int = 0):
        return self._linear().get(layer_idx, {}).get("rec")

    def is_recording_past(self, layer_idx: int) -> bool:
        return False

    def update_conv_state(self, conv_states: torch.Tensor, layer_idx: int, state_idx: int = 0,
                          conv_kernel_size: int | None = None, **kwargs) -> torch.Tensor:
        state = self._linear().setdefault(layer_idx, {"conv": None, "rec": None, "kernel": conv_kernel_size})
        kernel = conv_kernel_size or state["kernel"]
        state["kernel"] = kernel
        if state["conv"] is None:
            full = conv_states
            if full.shape[-1] < kernel:
                full = torch.nn.functional.pad(full, (kernel - full.shape[-1], 0), value=0)
        else:
            full = torch.cat([state["conv"], conv_states], dim=-1)
        state["conv"] = full[..., -kernel:].clone()
        return full

    def update_recurrent_state(self, recurrent_states: torch.Tensor, layer_idx: int, state_idx: int = 0, **kwargs):
        state = self._linear().setdefault(layer_idx, {"conv": None, "rec": None, "kernel": None})
        state["rec"] = recurrent_states.clone()
        return state["rec"]

    def snapshot_linear(self):
        """Keep a copy of the linear-attention states as they are now (e.g. right after the image prefix)."""
        self.linear_snapshot = {layer: {k: (v.clone() if torch.is_tensor(v) else v) for k, v in s.items()}
                                for layer, s in self._linear().items()}

    def retain(self, layer: int, slots: torch.Tensor):
        """Keep only the given slots (sorted) of a layer and compact the buffer."""
        count = slots.numel()
        self.k[layer][:, :, :count] = self.k[layer][:, :, slots]
        self.v[layer][:, :, :count] = self.v[layer][:, :, slots]
        self.index[layer][:count] = self.index[layer][slots]
        if self.head_mask[layer] is not None:
            self.head_mask[layer][:, :count] = self.head_mask[layer][:, slots]
        self.length[layer] = count
        self.selected[layer] = None

    def evict(self, layer: int, token_ids: torch.Tensor):
        """Physically remove the given original token indices from a layer."""
        keep = ~torch.isin(self.tokens(layer), token_ids)
        self.retain(layer, keep.nonzero().squeeze(1))

    def gather_heads(self, layer: int, slots: torch.Tensor):
        """Per-KV-head compaction: `slots` (H_kv, N') holds each head's kept slots, sorted, the same count per head."""
        count, n = slots.shape[1], self.length[layer]
        index = slots[None, :, :, None].expand(self.batch, -1, -1, self.k[layer].shape[3])
        self.k[layer][:, :, :count] = torch.gather(self.k[layer][:, :, :n], 2, index)
        self.v[layer][:, :, :count] = torch.gather(self.v[layer][:, :, :n], 2, index)
        self.index[layer][:count] = self.index[layer][slots[0]]
        if self.head_mask[layer] is not None:
            self.head_mask[layer][:, :count] = torch.gather(self.head_mask[layer][:, :n], 1, slots)
        self.length[layer] = count
        self.selected[layer] = None

    def mask_heads(self, layer: int, keep: torch.Tensor):
        """Per-KV-head eviction: `keep` (H_kv, N) bool over the layer's current slots."""
        if self.head_mask[layer] is None:
            self.head_mask[layer] = torch.ones(self.kv_heads, self.k[layer].shape[2], dtype=torch.bool,
                                               device=keep.device)
        self.head_mask[layer][:, :self.length[layer]] &= keep

    def select(self, layer: int, slots: torch.Tensor | None):
        """Restrict attention to `slots` (plus entries appended later) until the next `select`; None clears it."""
        self.selected[layer] = None if slots is None else (slots, self.length[layer])

    def fork(self, batch: int, extra: int) -> "KVCache":
        """A copy whose rows all start from this cache's (single-row) contents, with room for `extra` tokens."""
        layers = self.num_layers
        k0 = self.k[0]
        fork = KVCache(layers, batch, self.kv_heads, k0.shape[3], max(self.length) + extra, k0.dtype, k0.device)
        for layer in range(layers):
            n = self.length[layer]
            fork.k[layer][:, :, :n] = self.k[layer][:1, :, :n]
            fork.v[layer][:, :, :n] = self.v[layer][:1, :, :n]
            fork.index[layer][:n] = self.index[layer][:n]
            fork.length[layer] = n
            if self.head_mask[layer] is not None:
                fork.head_mask[layer] = torch.ones_like(fork.index[layer], dtype=torch.bool)[None].repeat(
                    self.kv_heads, 1)
                fork.head_mask[layer][:, :n] = self.head_mask[layer][:, :n]
        if getattr(self, "linear", None):   # linear-attention states: from the snapshot when one was taken
            source = self.linear_snapshot if self.linear_snapshot is not None else self.linear
            fork.linear, fork.linear_snapshot = {}, None
            for layer, s in source.items():
                fork.linear[layer] = {k: (v[:1].expand(batch, *v.shape[1:]).clone() if torch.is_tensor(v) else v)
                                      for k, v in s.items()}
        return fork

    def nbytes(self) -> int:
        """Bytes of the stored (not preallocated) K/V entries; head-masked entries count as freed."""
        k = self.k[0]
        per_entry = 2 * k.shape[0] * k.shape[3] * k.element_size()  # K and V of one head and token
        total = 0
        for layer, n in enumerate(self.length):
            mask = self.head_mask[layer]
            total += per_entry * (int(mask[:, :n].sum()) if mask is not None else k.shape[1] * n)
        return total

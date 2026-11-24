"""Budget and cost accounting: retained visual KV per layer and step, analytical FLOPs, wall-clock and memory.

Counts are accumulated as GPU tensors and read once at the end, so accounting adds no host syncs to decoding.
"""
import time
from dataclasses import dataclass, field

import torch

from ..models.base import DecoderSpec


@dataclass
class Costs:
    vision_seconds: float = 0.0
    prefill_seconds: float = 0.0     # prefill including input reduction and post-prefill compression
    decode_seconds: float = 0.0
    decode_steps: int = 0
    peak_memory_gb: float = 0.0
    kv_bytes_after_prefill: int = 0
    prefill_tflops: float = 0.0
    decode_tflops: float = 0.0
    retained_visual: float = 1.0     # mean over decode steps and layers of visual keys read / visual tokens
    retained_visual_by_layer: list = field(default_factory=list)

    @property
    def ttft(self) -> float:
        return self.vision_seconds + self.prefill_seconds

    @property
    def tpot(self) -> float:
        return self.decode_seconds / max(1, self.decode_steps)


class Accountant:
    def __init__(self, spec: DecoderSpec, visual_lookup: torch.Tensor, num_visual: int):
        self.spec, self.lookup, self.num_visual = spec, visual_lookup, max(1, num_visual)
        device = visual_lookup.device
        self.visual_keys = torch.zeros(spec.num_layers, device=device)  # summed over decode steps
        self.decode_steps = 0
        self.linear_flops = {"prefill": 0.0, "decode": 0.0}
        self.attention_flops = {"prefill": torch.zeros((), device=device), "decode": torch.zeros((), device=device)}

    def record(self, layer: int, phase: str, batch: int, queries: int, key_index: torch.Tensor,
               key_mask: torch.Tensor | None):
        if phase not in self.linear_flops:
            return
        spec = self.spec
        visual = self.lookup[key_index]  # the lookup covers prompt and generated token indices
        if key_mask is None:
            keys = torch.tensor(float(key_index.numel()), device=key_index.device)
            visual_keys = visual.sum().float()
        else:  # per-head masks: the average over KV heads of the keys each head still reads
            keys = key_mask[0].float().sum() / spec.num_kv_heads
            visual_keys = (key_mask[0] & visual).float().sum() / spec.num_kv_heads
        self.linear_flops[phase] += 2.0 * batch * queries * spec.params_per_layer
        # QK^T and AV; a causal prefill reads about half of the keys on average
        share = 0.5 if phase == "prefill" else 1.0
        self.attention_flops[phase] += 4.0 * batch * spec.num_heads * spec.head_dim * queries * keys * share
        if phase == "decode":
            self.visual_keys[layer] += visual_keys

    def step(self):
        self.decode_steps += 1

    def finish(self, costs: Costs) -> Costs:
        costs.prefill_tflops = (self.linear_flops["prefill"] + self.attention_flops["prefill"].item()) / 1e12
        costs.decode_tflops = (self.linear_flops["decode"] + self.attention_flops["decode"].item()) / 1e12
        if self.decode_steps:
            by_layer = (self.visual_keys / (self.decode_steps * self.num_visual)).tolist()
            costs.retained_visual_by_layer = by_layer
            costs.retained_visual = sum(by_layer) / len(by_layer)
        return costs


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    def __init__(self):
        synchronize()
        self.start = time.perf_counter()

    def lap(self) -> float:
        synchronize()
        now = time.perf_counter()
        elapsed, self.start = now - self.start, now
        return elapsed

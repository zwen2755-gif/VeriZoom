"""Greedy generation driving the compressor hooks, with optional teacher forcing for fidelity measurements."""
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..models.base import Prompt
from .forward import Decoder
from .profiler import Accountant, Costs, Timer


@dataclass
class Generation:
    tokens: torch.Tensor                    # (T,) generated (or teacher) tokens
    log_probs: torch.Tensor | None = None   # (T, V) float32 on the GPU when requested
    costs: Costs = field(default_factory=Costs)


class Engine:
    def __init__(self, adapter):
        self.adapter = adapter
        self.decoder = Decoder(adapter)

    @torch.no_grad()
    def run(self, prompt: Prompt, compressor=None, max_new_tokens: int = 64, teacher: torch.Tensor | None = None,
            keep_log_probs: bool = False, ignore_eos: bool = False, one_pass: bool = False) -> Generation:
        """Greedy decoding (stopping at EOS), or teacher-forced decoding of `teacher` when given. In both cases the
        compressor acts exactly as in deployment: it only ever sees the prefix decoded so far.

        `one_pass` (teacher forcing only): when the compressor's kept entries are final after the prefill, the whole
        teacher is fed in one causal forward pass instead of one step per token. The cache it attends to is the same,
        so the distributions match the stepwise replay up to floating-point reduction order. Compressors that act
        while decoding (DyCoke) always replay step by step."""
        if (one_pass and teacher is not None and len(teacher) > 1
                and (compressor is None or compressor.static_after_prefill)):
            return self.replay_one_pass(prompt, compressor, teacher, keep_log_probs)
        adapter, device = self.adapter, prompt.embeds.device
        steps = len(teacher) if teacher is not None else max_new_tokens
        lookup = torch.cat([prompt.visual_lookup(), torch.zeros(steps, dtype=torch.bool, device=device)])
        accountant = Accountant(self.decoder.spec, lookup, prompt.num_visual)
        costs = Costs(vision_seconds=prompt.vision_seconds)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        timer = Timer()
        if compressor is not None:
            compressor.reset(prompt, self.decoder.spec)
            prompt = compressor.reduce_inputs(prompt)
        cache = self.decoder.new_cache(1, prompt.length + steps)
        logits = self.decoder.forward(prompt.embeds, prompt.token_index, prompt.positions, cache, compressor,
                                      "prefill", accountant=accountant)
        if compressor is not None:
            compressor.after_prefill(cache)
        costs.prefill_seconds = timer.lap()
        costs.kv_bytes_after_prefill = cache.nbytes()

        tokens, log_probs = [], []
        eos = torch.tensor(adapter.eos_ids, device=device)
        for step in range(steps):
            if keep_log_probs:
                log_probs.append(F.log_softmax(logits[0, -1], -1))
            token = teacher[step:step + 1] if teacher is not None else logits[0, -1].argmax(-1, keepdim=True)
            tokens.append(token)
            if step == steps - 1 or (teacher is None and not ignore_eos and bool(torch.isin(token, eos))):
                break
            if compressor is not None:
                compressor.before_decode_step(step + 1, cache)
            logits = self.decoder.forward(adapter.embed(token)[None], prompt.original_length + step + token.new_zeros(1),
                                          adapter.text_positions(prompt.next_position + step, 1), cache, compressor,
                                          "decode", step + 1, accountant=accountant)
            accountant.step()
        costs.decode_seconds = timer.lap()
        costs.decode_steps = accountant.decode_steps
        if torch.cuda.is_available():
            costs.peak_memory_gb = torch.cuda.max_memory_allocated() / 2 ** 30
        accountant.finish(costs)
        return Generation(torch.cat(tokens), torch.stack(log_probs) if keep_log_probs else None, costs)

    @torch.no_grad()
    def replay_one_pass(self, prompt: Prompt, compressor, teacher: torch.Tensor, keep_log_probs: bool) -> Generation:
        """Prefill with the compressor, then the teacher tokens (all but the last) as one chunk over the compressed
        cache; the chunk adds no hooks, as the compressor is static after the prefill."""
        adapter, steps = self.adapter, len(teacher)
        costs = Costs(vision_seconds=prompt.vision_seconds)
        timer = Timer()
        if compressor is not None:
            compressor.reset(prompt, self.decoder.spec)
            prompt = compressor.reduce_inputs(prompt)
        cache = self.decoder.new_cache(1, prompt.length + steps)
        logits = self.decoder.forward(prompt.embeds, prompt.token_index, prompt.positions, cache, compressor,
                                      "prefill")
        if compressor is not None:
            compressor.after_prefill(cache)
        costs.prefill_seconds = timer.lap()
        chunk = teacher[:-1]
        index = prompt.original_length + torch.arange(len(chunk), device=chunk.device)
        rest = self.decoder.forward(adapter.embed(chunk)[None], index,
                                    adapter.text_positions(prompt.next_position, len(chunk)), cache, None, "replay",
                                    all_logits=True)
        costs.decode_seconds = timer.lap()
        log_probs = None
        if keep_log_probs:
            log_probs = F.log_softmax(torch.cat([logits[0, -1:], rest[0]]), -1)
        return Generation(teacher, log_probs, costs)

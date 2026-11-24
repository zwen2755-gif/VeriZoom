"""Glance -> propose -> focus -> read: training-free cascade with the evidence location as a latent variable.

    p(a | I, q) = sum_z p(a | glance, crop(z), q) p(z | I, q)

Stage 0  Glance. One prefill of the 1024-token glance + question gives the answer p0, the KV cache G (kept up to the
         end of the image and reused by every later pass) and, from a few localization heads, the attention of each
         content word of the question over the glance tokens.
Stage 1  Propose. pi = sum over words of the head-mean word maps (each normalized to sum 1), a distribution over the
         glance grid. Candidates are its modes (NMS, radius --rad tokens); each candidate box is the mass box grown
         from the mode inside its (2 rad + 1)^2 window until it holds --alpha of the window's mass, clipped to a
         minimum side and an aspect ratio in [1/4, 4].
Stage 2  Focus. Each candidate is read on its own fork of G (glance image + crop + question): P1[k] = p(a | B_k) and
         the same heads' word attention, of which `share_k` is the part on the crop. Region weights w_k from the
         evidence (rule): share (w ~ share_k), ig (w ~ exp(H(p0) - H(P1[k]))), both (product) or uniform.
         Proposals come from pi, so pi / q cancels and pi plays no part in w (its use is the ablation uniform vs
         evidence). p1 = sum_k w_k P1[k]. Everything is stored so that exits and temperatures are set offline on the
         train split.
Stage 3  Read. The regions holding --alpha of w (smallest set) are read together after the glance (fork of G), each
         at tokens ~ w_k (total --tokens, at least 64). Boxes: the stage-1 boxes (refine=0) or the mass box of the
         focus attention inside each crop (refine=1); either way the read is a new pass, nothing stale is reused.

usage: fovea_cascade.py --dataset mmerwtrain --every 4 [--rules share ig both uniform] [--tag ""]
       -> runs/fovea/<dataset>/cascade<tag>.jsonl
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import fovea  # noqa: E402,F401
from vzcore.engine import attention as vz_attention  # noqa: E402
from vzcore.models import load_model  # noqa: E402
from fovea.canvas import PATCH, Piece, resized  # noqa: E402
from fovea_diag_layout import warp  # noqa: E402
from fovea_diag_readform import to_tokens  # noqa: E402
from fovea_label import LOADERS, Reader, question_text  # noqa: E402
from fovea_region import content_rows  # noqa: E402

HEADS = [(19, 8), (22, 13), (22, 10)]
FORM_KEY = {"sep": "r", "warp": "w", "solo": "s"}
WARP_CAPTION = "Magnified view of the whole image (regions of interest enlarged, layout preserved):"


class Heads:
    """attention of query `rows` (local to the call) to absolute key `slots` for HEADS -> (len(HEADS), R, S)"""

    def __init__(self, rows, slots):
        self.rows, self.slots, self.layer, self.out = rows, slots, 0, {}

    def on_layer(self, context):
        if context.phase != "prefill":
            return None
        for l, h in HEADS:
            if l == context.layer:  # the decoder's layer index (models with linear-attention layers skip some)
                self.out[(l, h)] = context.probabilities(self.rows)[0][h][:, self.slots].float()
        self.layer += 1
        return None

    def maps(self):
        return torch.stack([self.out[h] for h in HEADS])


def word_rows(prompt, after, tok):
    """absolute indices of the question's content words (fallback: the whole question line) after index `after`"""
    tokens = [tok.decode([i]) for i in prompt.input_ids[after + 1:].tolist()]
    rows = content_rows(tokens)
    if not rows:
        rows = []
        for i, t in enumerate(tokens[1:], 1):
            if "\n" in t:
                break
            rows.append(i)
    return [after + 1 + i for i in rows]


def word_maps(rec):
    """(words, S): head-mean of each word's attention, each head map normalized over the recorded slots"""
    m = rec.maps()
    m = m / m.sum(-1, keepdim=True).clamp_min(1e-12)
    m = m.mean(0)
    return m / m.sum(-1, keepdim=True)


def mass_box(pmap, r, c, alpha, rad=None):
    """grid box (r0, r1, c0, c1), inclusive, grown greedily from (r, c) until it holds alpha of the window's mass
    (window: (2 rad + 1)^2 around (r, c), or the whole map)."""
    R, C = pmap.shape
    lo_r, hi_r = (0, R - 1) if rad is None else (max(0, r - rad), min(R - 1, r + rad))
    lo_c, hi_c = (0, C - 1) if rad is None else (max(0, c - rad), min(C - 1, c + rad))
    target = alpha * float(pmap[lo_r:hi_r + 1, lo_c:hi_c + 1].sum())
    r0 = r1 = r
    c0 = c1 = c
    mass = float(pmap[r, c])
    while mass < target:
        options = []
        if r0 > lo_r:
            options.append((float(pmap[r0 - 1, c0:c1 + 1].sum()), "top"))
        if r1 < hi_r:
            options.append((float(pmap[r1 + 1, c0:c1 + 1].sum()), "bottom"))
        if c0 > lo_c:
            options.append((float(pmap[r0:r1 + 1, c0 - 1].sum()), "left"))
        if c1 < hi_c:
            options.append((float(pmap[r0:r1 + 1, c1 + 1].sum()), "right"))
        if not options:
            break
        gain, side = max(options)
        mass += gain
        if side == "top":
            r0 -= 1
        elif side == "bottom":
            r1 += 1
        elif side == "left":
            c0 -= 1
        else:
            c1 += 1
    return r0, r1, c0, c1


def clip_box(x0, y0, x1, y1, W, H, min_side, max_aspect=4.0):
    w, h = max(x1 - x0, min_side), max(y1 - y0, min_side)
    w, h = max(w, h / max_aspect), max(h, w / max_aspect)
    w, h = min(w, W), min(h, H)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    nx0, ny0 = min(max(cx - w / 2, 0), W - w), min(max(cy - h / 2, 0), H - h)
    return int(nx0), int(ny0), int(nx0 + w), int(ny0 + h)


def modes(pi, k, rad):
    p = pi.clone()
    out = []
    for _ in range(k):
        i = int(p.argmax())
        r, c = divmod(i, p.shape[1])
        out.append((r, c))
        p[max(0, r - rad):r + rad + 1, max(0, c - rad):c + rad + 1] = -1.0
    return out


def prefill_glance(decoder, prompt, cut, cache, rec):
    """Prefill the glance prompt into `cache` -> logits. Models with linear-attention layers (recurrent states, no
    per-token K/V) prefill the image prefix [0, cut) first and snapshot their states, so that forks of the cache start
    right after the image; `rec` (rows local to the call) then sees the rest. Other models: one pass, as always."""
    if getattr(decoder.adapter, "has_linear", False):
        decoder.forward(prompt.embeds[:, :cut], prompt.token_index[:cut], prompt.positions[..., :cut], cache, None,
                        "prefill")
        cache.snapshot_linear()
        if rec is not None:
            rec.rows = rec.rows - cut
        return decoder.forward(prompt.embeds[:, cut:], prompt.token_index[cut:], prompt.positions[..., cut:], cache,
                               rec, "prefill")
    return decoder.forward(prompt.embeds, prompt.token_index, prompt.positions, cache, rec, "prefill")


def entropy(p):
    return float(-(p * p.clamp_min(1e-12).log()).sum())


def caption_of(box, W, H):
    b = tuple(round(v * 1000) for v in (box[0] / W, box[1] / H, box[2] / W, box[3] / H))
    return f"Zoomed-in view of region [{', '.join(map(str, b))}] (0-1000 coordinates):"


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mmerwtrain")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--rad", type=int, default=3, help="NMS radius and proposal window, glance tokens")
    parser.add_argument("--alpha", type=float, default=0.8, help="mass kept by boxes and by the read region set")
    parser.add_argument("--cand-tokens", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=576, help="read budget over the selected regions")
    parser.add_argument("--min-side", type=int, default=336)
    parser.add_argument("--rules", nargs="+", default=["share", "ig", "both", "uniform"])
    parser.add_argument("--refine", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--forms", nargs="+", default=["sep"], help="sep | warp (after G) | solo (warp alone)")
    parser.add_argument("--levels", type=int, default=1, help="2: proposals from 2x2 tiles at 2x glance resolution")
    parser.add_argument("--rho", type=float, default=0.25, help="warp: gap scale relative to the regions")
    parser.add_argument("--tag", default="")
    parser.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--offload", nargs="*", default=[], help="embed / head / visual kept in host memory (7B)")
    parser.add_argument("--heads", nargs="*", default=[], help="layer:head ... (default: HEADS)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--warm", action="store_true", help="run the first sample once untimed before the loop")
    args = parser.parse_args()
    loader, data_root = LOADERS[args.dataset]
    samples = loader(ROOT / data_root)
    if args.limit:
        samples = samples[:args.limit]
    samples = samples[::args.every]
    if args.warm:                          # warm-up: the first sample runs twice, the first run is not written
        samples = samples[:1] + samples
    out = ROOT / f"runs/fovea/{args.dataset}/cascade{args.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    vz_attention.LOWER_RIGHT_CAUSAL = True
    if args.heads:
        HEADS[:] = [tuple(map(int, x.split(":"))) for x in args.heads]
    print("heads", HEADS, "model", args.model, "offload", args.offload, flush=True)
    adapter = load_model("qwen2_5_vl", model_id=str(ROOT / args.model), dtype="bfloat16",
                         **({"offload": args.offload} if args.offload else {}))
    reader = Reader(adapter)
    reader.canvas.cache_size = 24
    decoder = reader.decoder
    tok = adapter.processor.tokenizer

    def sync():
        torch.cuda.synchronize()
        return time.perf_counter()

    with out.open("w") as sink:
        for number, sample in enumerate(samples):
            image = Image.open(sample["image"]).convert("RGB")
            image.load()
            W, H = image.size
            text, correct, letters = question_text(sample)
            letter_ids = torch.tensor([tok.convert_tokens_to_ids(l) for l in letters], device=adapter.device)
            answer = letters.index(correct)
            reader.canvas.cache.clear()
            t0 = sync()
            # ---- stage 0: glance (KV kept), answer, word attention over the glance
            small = resized(image, 1024)
            R, C = small.height // PATCH, small.width // PATCH
            gprompt, gspans = reader.canvas.build([Piece(small)], text)
            cut = int(gspans[0].max()) + 1
            rows = word_rows(gprompt, cut - 1, tok)
            rec = Heads(torch.tensor(rows, device=adapter.device), gspans[0])
            gcache = decoder.new_cache(1, gprompt.length + 1)
            logits = prefill_glance(decoder, gprompt, cut, gcache, rec)
            p0 = torch.log_softmax(logits[0, -1].float(), -1)[letter_ids].softmax(0)
            pi = word_maps(rec).sum(0).view(R, C)
            pi = (pi / pi.sum()).cpu()
            t1 = sync()

            def fork(tail):
                """glance image + `tail` pieces + question on a copy of G -> (option probs, prompt, spans, Heads)"""
                prompt, spans = reader.canvas.build([Piece(small)] + tail, text, separate=True)
                assert torch.equal(prompt.input_ids[:cut], gprompt.input_ids[:cut])
                wrows = word_rows(prompt, int(spans[-1].max()), tok)
                r = Heads(torch.tensor(wrows, device=adapter.device) - cut, torch.cat(spans))
                cache = gcache.fork(1, prompt.length - cut + 1)
                cache.length = [cut] * len(cache.length)
                lg = decoder.forward(prompt.embeds[:, cut:], prompt.token_index[cut:], prompt.positions[..., cut:],
                                     cache, r, "prefill")
                probs = torch.log_softmax(lg[0, -1].float(), -1)[letter_ids].softmax(0)
                return probs, prompt, spans, r

            # ---- stage 1: proposals from pi
            def proposals(pmap, rad):
                PR, PC = pmap.shape
                out = []
                for r, c in modes(pmap, args.k, rad):
                    r0, r1, c0, c1 = mass_box(pmap, r, c, args.alpha, rad)
                    out.append(clip_box(c0 / PC * W, r0 / PR * H, (c1 + 1) / PC * W, (r1 + 1) / PR * H, W, H,
                                        args.min_side))
                return out

            boxes = proposals(pi, args.rad)
            boxes1, t_level2 = boxes, 0.0
            if args.levels == 2:
                # level 2: the image in 2 x 2 overlapping tiles, each at the glance's token count (2x the glance's
                # linear resolution) and read on its own fork of G; each word's attention share on the tile's tokens
                # is placed on a 2R x 2C grid (max over overlaps) -> pi2, proposals from pi2
                ts2 = sync()
                FR, FC = 2 * R, 2 * C
                pi2 = torch.zeros(FR, FC)
                ov = 0.125
                tiles = []
                for ty in range(2):
                    for tx in range(2):
                        x0 = max(0.0, (tx * 0.5 - ov / 2) * W)
                        y0 = max(0.0, (ty * 0.5 - ov / 2) * H)
                        tiles.append((int(x0), int(y0), int(min(W, x0 + (0.5 + ov / 2) * W)),
                                      int(min(H, y0 + (0.5 + ov / 2) * H))))
                tile_images = [to_tokens(image.crop(t), 1024) for t in tiles]
                reader.canvas.encode_missing(tile_images)
                for t, ti in zip(tiles, tile_images):
                    _, _, spans, r = fork([Piece(ti, caption_of(t, W, H))])
                    on = word_maps(r)[:, len(spans[0]):].sum(0).cpu()
                    tr, tc = ti.height // PATCH, ti.width // PATCH
                    on = on.view(tr, tc)
                    ys = ((t[1] + (torch.arange(tr) + 0.5) * (t[3] - t[1]) / tr) / H * FR).long().clamp(0, FR - 1)
                    xs = ((t[0] + (torch.arange(tc) + 0.5) * (t[2] - t[0]) / tc) / W * FC).long().clamp(0, FC - 1)
                    for i in range(tr):
                        pi2[ys[i], xs] = torch.maximum(pi2[ys[i], xs], on[i])
                pi2 = pi2 / pi2.sum()
                boxes = proposals(pi2, 2 * args.rad)
                t_level2 = sync() - ts2
            crops = [to_tokens(image.crop(b), args.cand_tokens) for b in boxes]
            reader.canvas.encode_missing(crops)
            t2 = sync()
            # ---- stage 2: focus, one fork of G per candidate
            P1, share, refined = [], [], []
            for b, crop in zip(boxes, crops):
                probs, prompt, spans, r = fork([Piece(crop, caption_of(b, W, H))])
                wm = word_maps(r)                                        # (words, glance + crop)
                n0 = len(spans[0])
                on_crop = wm[:, n0:]
                P1.append(probs)
                share.append(float(on_crop.sum(1).mean()))
                cr, cc = crop.height // PATCH, crop.width // PATCH
                cmap = on_crop.sum(0).view(cr, cc).cpu()
                rr, ccc = divmod(int(cmap.argmax()), cc)
                q0, q1, s0, s1 = mass_box(cmap, rr, ccc, args.alpha)
                sx, sy = (b[2] - b[0]) / cc, (b[3] - b[1]) / cr
                refined.append(clip_box(b[0] + s0 * sx, b[1] + q0 * sy, b[0] + (s1 + 1) * sx, b[1] + (q1 + 1) * sy,
                                        W, H, args.min_side))
            P1t = torch.stack(P1)
            H0 = entropy(p0)
            Hk = [entropy(p) for p in P1]
            t3 = sync()
            row = {"id": sample["id"], "category": sample["category"], "answer": answer, "size": [W, H],
                   "grid": [R, C], "p0": p0.tolist(), "boxes": [list(b) for b in boxes],
                   "refined": [list(b) for b in refined], "boxes1": [list(b) for b in boxes1], "P1": P1t.tolist(), "share": share, "H0": H0, "Hk": Hk,
                   "seconds": {"glance": t1 - t0, "propose": t2 - t1, "focus": t3 - t2, "level2": t_level2}, "read": {}}
            # ---- stage 3: read the regions holding alpha of w, after the glance
            memo = {}
            for rule in args.rules:
                if rule == "share":
                    w = torch.tensor(share)
                elif rule == "ig":
                    w = torch.tensor([math.exp(H0 - h) for h in Hk])
                elif rule == "both":
                    w = torch.tensor([s * math.exp(H0 - h) for s, h in zip(share, Hk)])
                else:
                    w = torch.ones(len(boxes))
                w = w / w.sum()
                p1 = (w[:, None] * P1t.cpu()).sum(0)
                order = w.argsort(descending=True).tolist()
                chosen, mass = [], 0.0
                for j in order:
                    chosen.append(j)
                    mass += float(w[j])
                    if mass >= args.alpha - 1e-6:
                        break
                chosen.sort()
                total = float(w[chosen].sum())
                toks = [max(64, round(args.tokens * float(w[j]) / total)) for j in chosen]
                entry = {"w": w.tolist(), "p1": p1.tolist(), "chosen": chosen, "tokens": toks}
                for refine in args.refine:
                    src = refined if refine else boxes
                    for form in args.forms:
                        key = (form,) + tuple((tuple(src[j]), t) for j, t in zip(chosen, toks))
                        if form != "sep":
                            key = (form,) + tuple(tuple(src[j]) for j in chosen)
                        if key not in memo:
                            ts = sync()
                            if form == "sep":
                                tail = [Piece(to_tokens(image.crop(src[j]), t), caption_of(src[j], W, H))
                                        for j, t in zip(chosen, toks)]
                                reader.canvas.encode_missing([p.image for p in tail])
                                probs, prompt, _, _ = fork(tail)
                            elif form == "warp":
                                probs, prompt, _, _ = fork([Piece(warp(image, [src[j] for j in chosen], args.tokens,
                                                                       args.rho), WARP_CAPTION)])
                            else:
                                lp, prompt, _ = reader.log_probs(
                                    [Piece(warp(image, [src[j] for j in chosen], 1024 + args.tokens, args.rho))], text)
                                probs = lp.float()[letter_ids].softmax(0)
                            memo[key] = {"probs": probs.tolist(), "correct": int(probs.argmax()) == answer,
                                         "visual": prompt.num_visual, "seconds": sync() - ts}
                        entry[f"{FORM_KEY[form]}{refine}"] = memo[key]
                row["read"][rule] = entry
            row["seconds"]["total_first"] = t3 - t0 + entry_seconds(row, args)
            if args.warm and number == 0:
                continue
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            first = row["read"][args.rules[0]]
            print(f"[{number + 1}/{len(samples)}] {sample['id']} p0={'ok' if int(p0.argmax()) == answer else 'x'} "
                  + " ".join(f"{rule}:" + "".join("o" if row['read'][rule][f'{FORM_KEY[f]}{x}']['correct'] else "x"
                                                 for x in args.refine for f in args.forms) for rule in args.rules)
                  + f" t={row['seconds']['total_first']:.2f}", flush=True)


def entry_seconds(row, args):
    """seconds of the first rule's first refine read (the cost of one full cascade run)"""
    return row["read"][args.rules[0]][f"{FORM_KEY[args.forms[0]]}{args.refine[0]}"]["seconds"]


if __name__ == "__main__":
    main()

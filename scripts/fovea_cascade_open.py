"""Open-ended VeriZoom: every stage generates a short answer instead of reading option probabilities.

glance   prefill G, generate a0 with confidence c0 (geometric mean of its token probabilities), first-token entropy H0,
         proposal map pi from the question words (same heads and proposals as fovea_cascade.py)
verify   per candidate, on a fork of G: attention share s_k, first-token entropy H_k, generated answer a_k with c_k;
         weights w_k ~ s_k exp(H0 - H_k); verified answer a1 = the answer with the largest sum of w_k c_k, c1 = that sum
read     the regions holding alpha of w, budget split by w (separate crops or one warped image), generated a2, c2
Exits are chosen offline (fovea_open_select.py). Baselines in the same run: the whole image at fixed token counts and
ViCrop (relative attention over the 1,024-token image, top tile at 576 tokens after a 256-token overview), as in
fovea_e2e.py. Answers are scored by fovea_open.match against the text of the correct option.

    python scripts/fovea_cascade_open.py --dataset vstar --heads 24:5 22:10 22:0 --methods ours native vicrop
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
sys.path.insert(0, str(ROOT / "scripts"))
import fovea_cascade as FC  # noqa: E402
from fovea_cascade import (HEADS, PATCH, WARP_CAPTION, Heads, Piece, Reader, caption_of, clip_box, load_model,  # noqa: E402
                           mass_box, modes, resized, to_tokens, warp, word_maps, word_rows)
from fovea_e2e import GENERIC, OVERVIEW, pick, views_of, zoom_piece  # noqa: E402
from fovea.views import GlanceRecorder, token_owner  # noqa: E402
from fovea_open import answerable, match, norm, open_samples  # noqa: E402


def entropy_full(logits):
    p = torch.softmax(logits.float(), -1)
    return float(-(p * p.clamp_min(1e-12).log()).sum())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="vstar")
    ap.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--offload", nargs="*", default=[])
    ap.add_argument("--heads", nargs="*", default=["24:5", "22:10", "22:0"])
    ap.add_argument("--methods", nargs="+", default=["ours", "native", "vicrop"])
    ap.add_argument("--native", type=int, nargs="*", default=[1600, 8192])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--rad", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--cand-tokens", type=int, default=256)
    ap.add_argument("--tokens", type=int, default=576)
    ap.add_argument("--min-side", type=int, default=336)
    ap.add_argument("--forms", nargs="+", default=["sep", "warp"])
    ap.add_argument("--rho", type=float, default=0.25)
    ap.add_argument("--max-new", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")  # answers can hold characters the Windows console cannot print
    HEADS[:] = [tuple(map(int, x.split(":"))) for x in args.heads]
    FC.vz_attention.LOWER_RIGHT_CAUSAL = True
    # subsample the full list first, then drop the label-taxonomy questions: the same questions for every model
    samples = [s for s in open_samples(args.dataset, exclude_labels=False)[::args.every] if answerable(s)]
    if args.limit:
        samples = samples[:args.limit]
    out = ROOT / f"runs/fovea/{args.dataset}/open{args.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")}
    print("heads", HEADS, "model", args.model, "samples", len(samples), "done", len(done), flush=True)
    adapter = load_model("qwen2_5_vl", model_id=str(ROOT / args.model), dtype="bfloat16",
                         **({"offload": args.offload} if args.offload else {}))
    reader = Reader(adapter)
    reader.canvas.cache_size = 24
    decoder, tok = reader.decoder, adapter.processor.tokenizer
    eos = set(adapter.eos_ids)

    def sync():
        torch.cuda.synchronize()
        return time.perf_counter()

    def generate(prompt, cache, logits):
        """greedy answer continuing `prompt` in `cache` from the prefill logits -> (text, confidence, tokens)"""
        toks, lps = [], []
        for step in range(args.max_new):
            lp = torch.log_softmax(logits[0, -1].float(), -1)
            t = int(lp.argmax())
            if t in eos:
                break
            toks.append(t)
            lps.append(float(lp[t]))
            x = torch.tensor([t], device=adapter.device)
            logits = decoder.forward(adapter.embed(x)[None], prompt.original_length + step + x.new_zeros(1),
                                     adapter.text_positions(prompt.next_position + step, 1), cache, None, "decode",
                                     step + 1)
        text = tok.decode(toks, skip_special_tokens=True).strip()
        return text, (math.exp(sum(lps) / len(lps)) if lps else 0.0), len(toks)

    def answer_whole(pieces, text):
        """one fresh prefill of `pieces` + text and a generated answer"""
        prompt, _ = reader.canvas.build(pieces, text)
        cache = decoder.new_cache(1, prompt.length + 1 + args.max_new)
        logits = decoder.forward(prompt.embeds, prompt.token_index, prompt.positions, cache, None, "prefill")
        a, c, n = generate(prompt, cache, logits)
        return a, c, prompt.num_visual

    with out.open("a", encoding="utf-8") as sink:
        for number, s in enumerate(samples):
            if s["id"] in done:
                continue
            image = Image.open(s["image"]).convert("RGB")
            image.load()
            W, H = image.size
            text, gold, others = s["open_text"], s["gold"], s["others"]
            ok = lambda a: match(a, gold, others)  # noqa: E731
            row = {"id": s["id"], "category": s.get("category"), "gold": gold, "others": others, "size": [W, H]}
            reader.canvas.cache.clear()

            if "native" in args.methods:
                for n in args.native:
                    t0 = sync()
                    a, c, nv = answer_whole([Piece(resized(image, n))], text)
                    row[f"native{n}"] = {"answer": a, "conf": c, "correct": ok(a), "visual": nv, "seconds": sync() - t0}
                reader.canvas.cache.clear()

            if "vicrop" in args.methods:
                t0 = sync()
                small = resized(image, 1024)

                def attention(prompt_text):
                    prompt, spans = reader.canvas.build([Piece(small)], prompt_text)
                    last = int(spans[-1].max())
                    rec = GlanceRecorder(spans[0], torch.arange(last + 1, prompt.length, device=prompt.embeds.device))
                    decoder.forward(prompt.embeds, prompt.token_index, prompt.positions,
                                    decoder.new_cache(1, prompt.length + 1), rec, "prefill")
                    return rec.per_token()[:, 0].float()
                ratio = (attention(text) / (attention(GENERIC) + 1e-6)).mean(0).cpu()
                views = views_of(W, H, 576, 0.25, [1.0])
                owner = token_owner(views, small.height // PATCH, small.width // PATCH).float()
                score = owner @ ratio / owner.sum(1).clamp_min(1)
                i = sorted(pick(score, [v.box for v in views], 1))[0]
                a, c, nv = answer_whole([Piece(resized(image, 256), OVERVIEW), zoom_piece(image, views[i], 576)], text)
                row["vicrop"] = {"answer": a, "conf": c, "correct": ok(a), "visual": nv, "seconds": sync() - t0,
                                 "view": list(views[i].pixels)}
                reader.canvas.cache.clear()

            if "ours" in args.methods:
                t0 = sync()
                small = resized(image, 1024)
                R, C = small.height // PATCH, small.width // PATCH
                gprompt, gspans = reader.canvas.build([Piece(small)], text)
                cut = int(gspans[0].max()) + 1
                rec = Heads(torch.tensor(word_rows(gprompt, cut - 1, tok), device=adapter.device), gspans[0])
                gcache = decoder.new_cache(1, gprompt.length + 1 + args.max_new)
                logits = FC.prefill_glance(decoder, gprompt, cut, gcache, rec)
                H0 = entropy_full(logits[0, -1])
                pi = word_maps(rec).sum(0).view(R, C)
                pi = (pi / pi.sum()).cpu()
                a0, c0, _ = generate(gprompt, gcache, logits)
                t1 = sync()

                def fork(tail):
                    prompt, spans = reader.canvas.build([Piece(small)] + tail, text, separate=True)
                    assert torch.equal(prompt.input_ids[:cut], gprompt.input_ids[:cut])
                    wrows = word_rows(prompt, int(spans[-1].max()), tok)
                    r = Heads(torch.tensor(wrows, device=adapter.device) - cut, torch.cat(spans))
                    cache = gcache.fork(1, prompt.length - cut + 1 + args.max_new)
                    cache.length = [cut] * len(cache.length)
                    lg = decoder.forward(prompt.embeds[:, cut:], prompt.token_index[cut:], prompt.positions[..., cut:],
                                         cache, r, "prefill")
                    hk = entropy_full(lg[0, -1])
                    a, c, _ = generate(prompt, cache, lg)
                    return a, c, hk, prompt, spans, r

                boxes = []
                for r_, c_ in modes(pi, args.k, args.rad):
                    r0, r1, q0, q1 = mass_box(pi, r_, c_, args.alpha, args.rad)
                    boxes.append(clip_box(q0 / C * W, r0 / R * H, (q1 + 1) / C * W, (r1 + 1) / R * H, W, H,
                                          args.min_side))
                crops = [to_tokens(image.crop(b), args.cand_tokens) for b in boxes]
                reader.canvas.encode_missing(crops)
                t2 = sync()
                cand = []
                for b, crop in zip(boxes, crops):
                    a, c, hk, prompt, spans, r = fork([Piece(crop, caption_of(b, W, H))])
                    on_crop = word_maps(r)[:, len(spans[0]):]
                    cand.append({"answer": a, "conf": c, "H": hk, "share": float(on_crop.sum(1).mean()),
                                 "correct": ok(a)})
                t3 = sync()
                row["ours"] = {"a0": a0, "c0": c0, "H0": H0, "ok0": ok(a0), "boxes": [list(b) for b in boxes],
                               "cand": cand, "grid": [R, C], "read": {},
                               "seconds": {"glance": t1 - t0, "propose": t2 - t1, "verify": t3 - t2}}
                for rule in ("both", "uniform"):
                    if rule == "both":
                        w = torch.tensor([x["share"] * math.exp(H0 - x["H"]) for x in cand])
                    else:
                        w = torch.ones(len(cand))
                    w = w / w.sum()
                    pool = {}
                    for x, wk in zip(cand, w.tolist()):
                        key = norm(x["answer"])
                        if key:
                            v = pool.setdefault(key, [0.0, x["answer"]])
                            v[0] += wk * x["conf"]
                    key = max(pool, key=lambda k: pool[k][0]) if pool else ""
                    a1, c1 = (pool[key][1], pool[key][0]) if pool else ("", 0.0)
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
                    entry = {"w": w.tolist(), "a1": a1, "c1": c1, "ok1": ok(a1), "chosen": chosen, "tokens": toks}
                    if rule == "both":
                        for form in args.forms:
                            ts = sync()
                            if form == "sep":
                                tail = [Piece(to_tokens(image.crop(boxes[j]), t), caption_of(boxes[j], W, H))
                                        for j, t in zip(chosen, toks)]
                                reader.canvas.encode_missing([p.image for p in tail])
                            else:
                                tail = [Piece(warp(image, [boxes[j] for j in chosen], args.tokens, args.rho),
                                              WARP_CAPTION)]
                            a2, c2, _, prompt, _, _ = fork(tail)
                            entry[form] = {"a2": a2, "c2": c2, "ok2": ok(a2), "visual": prompt.num_visual,
                                           "seconds": sync() - ts}
                    row["ours"]["read"][rule] = entry

            sink.write(json.dumps(row) + "\n")
            sink.flush()
            o = row.get("ours", {})
            print(f"[{number + 1}/{len(samples)}] {s['id']} gold={gold!r} "
                  + (f"a0={o['a0']!r}{'+' if o['ok0'] else '-'} a1={o['read']['both']['a1']!r} "
                     f"a2={o['read']['both'][args.forms[0]]['a2']!r}{'+' if o['read']['both'][args.forms[0]]['ok2'] else '-'} "
                     if o else "")
                  + " ".join(f"{k}={row[k]['answer']!r}{'+' if row[k]['correct'] else '-'}"
                             for k in row if k.startswith("native") or k == "vicrop"), flush=True)


if __name__ == "__main__":
    main()

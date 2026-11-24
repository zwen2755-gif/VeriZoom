"""End-to-end FOVEA on images with a saved forecaster (fovea_analyze.py --save-model), timed stage by stage:

  glance    the image at --glance-tokens with the question, one prefill that records the glance signals
  features  view features (depth-binned attention / K/V norms, detail, geometry) for every view at every scale
  forecast  the saved ridge forecaster scores the views; non-maximum suppression picks K
  read      glance + the K views at --tile-tokens each, one prefill -> answer
  verify-M  (optional) M single-view reads, the lowest-entropy answer is kept

Compared with the native image (--ref-tokens). Everything after the image is decoded is inside the timer.

usage: fovea_e2e.py --dataset hrbench8k --forecaster runs/fovea/hrbench4k/forecaster.pt [--k 1 2 4] [--verify 3]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
Image.MAX_IMAGE_PIXELS = None

import fovea  # noqa: E402,F401
from vzcore.models import load_model  # noqa: E402
from vzcore.engine import attention as vz_attention  # noqa: E402
from fovea.canvas import PATCH, Piece, resized  # noqa: E402
from fovea.views import GlanceDone, GlanceRecorder, View, depth_bins, detail_fast, tile_views, token_owner, view_features, work_copy  # noqa: E402
from fovea_label import LOADERS, Reader, question_text  # noqa: E402
from fovea_policy import pick  # noqa: E402
from fovea_analyze_glance import glance_features  # noqa: E402

OVERVIEW = "Overview of the whole image:"
GENERIC = "Write a general description of the image."


def views_of(width, height, tile_tokens, overlap, scales):
    views, seen = [], set()
    for scale in scales:
        for view in tile_views(width, height, tile_tokens, overlap, scale):
            if view.pixels not in seen:
                seen.add(view.pixels)
                views.append(view)
    return views


def shared_glance(reader, piece, texts, make_recorder):
    """Prefill the prompts [piece] + text for every text, sharing the prefix up to the last image token: the system
    turn and the image are prefilled once (causal attention: identical for every text), then each text suffix runs on
    a copy of that cache with its own recorder. -> [(log_probs, prompt, recorder)] in the order of `texts`."""
    built = [reader.canvas.build([piece], text) for text in texts]
    cut = int(built[0][1][-1].max()) + 1                                 # one past the last image token
    first = built[0][0]
    for prompt, _ in built[1:]:
        assert torch.equal(prompt.input_ids[:cut], first.input_ids[:cut])
    decoder = reader.decoder
    base = decoder.new_cache(1, cut)
    decoder.forward(first.embeds[:, :cut], first.token_index[:cut], first.positions[..., :cut], base, None, "prefill")
    out = []
    for prompt, spans in built:
        recorder = make_recorder(prompt, spans)
        recorder.question_rows = recorder.question_rows - cut            # rows are local to the suffix call
        cache = base.fork(1, prompt.length - cut + 1)
        logits = decoder.forward(prompt.embeds[:, cut:], prompt.token_index[cut:], prompt.positions[..., cut:], cache,
                                 recorder, "prefill")
        out.append((torch.log_softmax(logits[0, -1], -1), prompt, recorder))
    return out


def peak_views(signal, rows, cols, width, height, tile_tokens, k):
    """Anchor-free views (FCOS-style centres): windows of the tile footprint centred on the k strongest glance tokens,
    a new centre only when it lies outside half a window of every earlier centre; windows are clamped into the
    image. Returned in reading (raster) order."""
    side = min(int((tile_tokens ** 0.5) * PATCH), max(width, height))
    sw, sh = min(side, width), min(side, height)
    centres = []
    for index in signal.argsort(descending=True).tolist():
        y, x = divmod(index, cols)
        cx, cy = (x + 0.5) / cols * width, (y + 0.5) / rows * height
        if all(abs(cx - ox) >= sw / 2 or abs(cy - oy) >= sh / 2 for ox, oy in centres):
            centres.append((cx, cy))
        if len(centres) == k:
            break
    views = []
    for cx, cy in centres:
        x0 = int(min(max(cx - sw / 2, 0), width - sw))
        y0 = int(min(max(cy - sh / 2, 0), height - sh))
        views.append(View((x0 / width, y0 / height, (x0 + sw) / width, (y0 + sh) / height), (x0, y0, x0 + sw, y0 + sh)))
    return sorted(views, key=lambda v: (v.pixels[1], v.pixels[0]))


def zoom_piece(image, view, tokens):
    x0, y0, x1, y1 = (round(v * 1000) for v in view.box)
    return Piece(resized(view.crop(image), tokens), f"Zoomed-in view of region [{x0}, {y0}, {x1}, {y1}] "
                                                    "(0-1000 coordinates):")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="vstar")
    parser.add_argument("--forecaster", required=True)
    parser.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--k", type=int, nargs="*", default=[1, 2, 4])
    parser.add_argument("--verify", type=int, nargs="*", default=[2, 3])
    parser.add_argument("--scales", type=float, nargs="+", default=[1, 2, 4])
    parser.add_argument("--glance-tokens", type=int, default=256)
    parser.add_argument("--overview-tokens", type=int, default=0,
                        help="overview inside every read (default: --glance-tokens); 1024 = the feature glance image")
    parser.add_argument("--peaks", action="store_true",
                        help="with a head selector: also read anchor-free windows centred on the head's peak tokens")
    parser.add_argument("--peak-smooth", type=float, default=0.25, help="centredness box, fraction of the window")
    parser.add_argument("--tile-tokens", type=int, default=576)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--ref-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--every", type=int, default=1, help="every n-th sample (latency subsets)")
    parser.add_argument("--tag", default="")
    parser.add_argument("--no-downscale", action="store_true", help="skip the whole image resized to each budget")
    parser.add_argument("--no-reference", action="store_true", help="skip the native-image reference read")
    parser.add_argument("--shared-glance", action="store_true",
                        help="one shared image prefill + two text suffixes (identical signals, but measured slower)")
    parser.add_argument("--reuse-glance", action="store_true",
                        help="every read continues the question glance's KV cache: the glance image is the overview "
                             "(no caption, encoded once), only the zoom tiles and the question are prefilled")
    parser.add_argument("--vicrop", type=int, nargs="*", default=[],
                        help="ViCrop-style relative attention at these glance sizes, same overview + K crops")
    args = parser.parse_args()

    vz_attention.LOWER_RIGHT_CAUSAL = args.reuse_glance
    model = torch.load(ROOT / args.forecaster)
    loader, data_root = LOADERS[args.dataset]
    samples = loader(ROOT / data_root)
    if args.limit:
        samples = samples[:args.limit]
    samples = samples[::args.every]
    out = ROOT / f"runs/fovea/{args.dataset}/e2e{args.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    from collections import Counter
    done, seen = Counter(), Counter()
    if out.exists():
        # MME-RealWorld-Lite repeats some ids: resume by (id, occurrence), the file is written in sample order
        done = Counter(r["id"] for r in map(json.loads, out.read_text().splitlines()))
    adapter = load_model("qwen2_5_vl", model_id=str(ROOT / args.model), dtype="bfloat16")
    reader = Reader(adapter)
    reader.canvas.cache_size = 8 if args.reuse_glance else 4   # glances share one vision encoding (reuse: + reads)
    tokenizer = adapter.processor.tokenizer

    def clock():
        torch.cuda.synchronize()
        return time.perf_counter()

    # warm-up (kernels, allocator) so that the first sample is not charged for it
    warm = Image.new("RGB", (1024, 1024), "gray")
    reader.log_probs([Piece(resized(warm, 1024))], "Describe.")

    with out.open("a") as sink:
        for number, sample in enumerate(samples):
            seen[sample["id"]] += 1
            if seen[sample["id"]] <= done[sample["id"]]:
                continue
            image = Image.open(sample["image"]).convert("RGB")
            image.load()
            text, correct, letters = question_text(sample)
            letter_ids = torch.tensor([tokenizer.convert_tokens_to_ids(l) for l in letters], device=adapter.device)
            answer = letters.index(correct)
            row = {"id": sample["id"], "category": sample["category"], "size": image.size}

            if not args.no_reference:
                start = clock()
                lp, prompt, _ = reader.log_probs([Piece(resized(image, args.ref_tokens))], text)
                row["reference"] = {"correct": int(lp[letter_ids].argmax()) == answer, "tokens": prompt.num_visual,
                                    "seconds": clock() - start, "probs": lp[letter_ids].softmax(0).tolist()}

            start = clock()
            glance_image = resized(image, args.glance_tokens)             # detail scale (and the old overview)
            overview_image = resized(image, args.overview_tokens) if args.overview_tokens else glance_image

            def make_recorder(prompt, spans):
                last = int(spans[-1].max())
                return GlanceRecorder(spans[0], torch.arange(last + 1, prompt.length, device=prompt.embeds.device),
                                      per_head=bool(model.get("heads")))

            vicrop_scores = None
            if model.get("glance"):
                # feature glance: the image at model["glance"] tokens with the question (and a generic prompt for
                # the relative-attention features); the same two passes also give the ViCrop-style ranking
                small = resized(image, model["glance"])
                if model.get("layers"):
                    # lite glance: question only, the prefill stops after model["layers"] decoder layers
                    prompt, spans = reader.canvas.build([Piece(small)], text)
                    last = int(spans[-1].max())
                    recorder = GlanceRecorder(spans[0], torch.arange(last + 1, prompt.length, device=prompt.embeds.device),
                                              stop_after=model["layers"])
                    try:
                        reader.decoder.forward(prompt.embeds, prompt.token_index, prompt.positions,
                                               reader.decoder.new_cache(1, prompt.length + 1), recorder, "prefill")
                    except GlanceDone:
                        pass
                    question = generic = recorder.per_token().float().cpu()
                    t_lite = clock() - start
                    # ViCrop (and the glance-only answer): full question + generic passes, timed on their own with
                    # their own vision encoding and resizes
                    reader.canvas.cache.clear()
                    vstart = clock()
                    vsmall, _ = resized(image, model["glance"]), resized(image, args.glance_tokens)
                    glance_lp, glance_prompt, rq = reader.log_probs([Piece(vsmall)], text, make_recorder)
                    _, _, rg = reader.log_probs([Piece(vsmall)], GENERIC, make_recorder)
                    vq, vg = rq.per_token()[:, 0].float().cpu(), rg.per_token()[:, 0].float().cpu()
                    t_vicrop = clock() - vstart
                    start = clock() - t_lite                              # the glance timer excludes ViCrop
                elif args.reuse_glance:
                    glance_prompt, spans = reader.canvas.build([Piece(small)], text)
                    recorder = make_recorder(glance_prompt, spans)
                    glance_cache = reader.decoder.new_cache(1, glance_prompt.length + 1)
                    logits = reader.decoder.forward(glance_prompt.embeds, glance_prompt.token_index,
                                                    glance_prompt.positions, glance_cache, recorder, "prefill")
                    glance_lp = torch.log_softmax(logits[0, -1], -1)
                    glance_cut = int(spans[0].max()) + 1                  # one past the glance image's last token
                    overview_image = small
                    question = recorder.per_token().float().cpu()
                    generic = question
                    if model["ratio"]:
                        _, _, generic_rec = reader.log_probs([Piece(small)], GENERIC, make_recorder)
                        generic = generic_rec.per_token().float().cpu()
                    if model.get("heads"):
                        heads_idx = torch.tensor(model["heads"])
                        hq = recorder.per_head()[heads_idx[:, 0], heads_idx[:, 1]].float().cpu()      # (n, Ng)
                        hg = generic_rec.per_head()[heads_idx[:, 0], heads_idx[:, 1]].float().cpu()
                elif not args.shared_glance:
                    glance_lp, glance_prompt, recorder = reader.log_probs([Piece(small)], text, make_recorder)
                    question = recorder.per_token().float().cpu()
                    generic = question
                    if model["ratio"]:
                        _, _, generic_rec = reader.log_probs([Piece(small)], GENERIC, make_recorder)
                        generic = generic_rec.per_token().float().cpu()
                    if model.get("heads"):
                        heads_idx = torch.tensor(model["heads"])
                        hq = recorder.per_head()[heads_idx[:, 0], heads_idx[:, 1]].float().cpu()      # (n, Ng)
                        hg = generic_rec.per_head()[heads_idx[:, 0], heads_idx[:, 1]].float().cpu()
                else:
                    passes = shared_glance(reader, Piece(small), [text, GENERIC] if model["ratio"] else [text],
                                           make_recorder)
                    glance_lp, glance_prompt, recorder = passes[0]
                    question = recorder.per_token().float().cpu()
                    generic = passes[-1][2].per_token().float().cpu()
                rows, cols = small.height // PATCH, small.width // PATCH
            else:
                glance_lp, glance_prompt, recorder = reader.log_probs([Piece(glance_image, OVERVIEW)], text,
                                                                      make_recorder)
                question = generic = recorder.per_token().float().cpu()
                rows, cols = glance_image.height // PATCH, glance_image.width // PATCH
            t_glance = clock() - start

            def read(tail):
                """overview + zoom pieces `tail` + question -> (log_probs, visual tokens in context, tokens prefilled)"""
                if not args.reuse_glance:
                    lp, prompt, _ = reader.log_probs([Piece(overview_image, OVERVIEW)] + tail, text)
                    return lp, prompt.num_visual, prompt.length
                prompt, _ = reader.canvas.build([Piece(overview_image)] + tail, text, separate=True)
                assert torch.equal(prompt.input_ids[:glance_cut], glance_prompt.input_ids[:glance_cut])
                cache = glance_cache.fork(1, prompt.length - glance_cut + 1)
                cache.length = [glance_cut] * len(cache.length)          # drop the glance question, keep the image
                logits = reader.decoder.forward(prompt.embeds[:, glance_cut:], prompt.token_index[glance_cut:],
                                                prompt.positions[..., glance_cut:], cache, None, "prefill")
                return torch.log_softmax(logits[0, -1], -1), prompt.num_visual, prompt.length - glance_cut

            start = clock()
            views = views_of(image.width, image.height, args.tile_tokens, args.overlap, model.get("scales", args.scales))
            owner = token_owner(views, rows, cols)
            boxes_list = [v.box for v in views]
            if model.get("heads"):
                # head selector: per chosen head, log share of the view's question / generic attention (summed over
                # the view's glance tokens), z-scored across views and averaged over the heads; no forecaster
                of = owner.float()
                share = ((of @ hq.T + 1e-9) / (of @ hg.T + 1e-9)).log()                       # (V, n)
                head_scores = ((share - share.mean(0)) / (share.std(0) + 1e-6)).mean(1)
                vicrop_scores = of @ (question[:, 0] / (generic[:, 0] + 1e-6)).mean(0) / of.sum(1).clamp_min(1)
                x = None
            elif model.get("glance"):
                detail = detail_fast(image, views, glance_image.width / image.width)
                x = glance_features(question, generic, owner, boxes_list, detail, model["bins"], model["ratio"])
                if model.get("layers"):
                    of = owner.float()
                    vicrop_scores = of @ (vq / (vg + 1e-6)).mean(0) / of.sum(1).clamp_min(1)
                elif model["ratio"]:
                    of = owner.float()
                    vicrop_scores = of @ (question[:, 0] / (generic[:, 0] + 1e-6)).mean(0) / of.sum(1).clamp_min(1)
            else:
                detail = detail_fast(image, views, glance_image.width / image.width)
                x = glance_features(question, generic, owner, boxes_list, detail, model["bins"], ratio=False)
            t_features = clock() - start

            start = clock()
            if x is None:
                scores = head_scores
            else:
                z = torch.cat([(x - model["mean"]) / model["std"], torch.ones(len(x), 1)], 1)
                scores = z @ model["w"]
            t_forecast = clock() - start
            row.update({"views": len(views), "glance": {"correct": int(glance_lp[letter_ids].argmax()) == answer,
                        "tokens": glance_prompt.num_visual, "seconds": t_glance,
                        "probs": glance_lp[letter_ids].softmax(0).tolist()},
                        "t_features": t_features, "t_forecast": t_forecast})
            pre = t_glance + t_features + t_forecast
            pres = {"fovea": pre, "vicrop": pre}
            if model.get("layers"):
                pres["vicrop"] = t_vicrop + t_features                    # its own two full passes + view pooling
                row["vicrop_pre"] = pres["vicrop"]

            boxes_list = [v.box for v in views]
            selectors = {"fovea": scores}
            if vicrop_scores is not None:
                selectors["vicrop"] = vicrop_scores
            for name, ranking in selectors.items():
                for k in args.k:
                    start = clock()
                    chosen = sorted(pick(ranking, boxes_list, k))
                    lp, ntok, nfill = read([zoom_piece(image, views[i], args.tile_tokens) for i in chosen])
                    t_read = clock() - start
                    row[f"{name}-{k}"] = {"correct": int(lp[letter_ids].argmax()) == answer,
                                          "tokens": ntok, "prefill": nfill, "seconds": pres[name] + t_read, "read": t_read,
                                          "views": chosen, "probs": lp[letter_ids].softmax(0).tolist()}
            if model.get("heads") and args.peaks:
                start = clock()
                ratio_tok = hq / (hg + 1e-6)
                peak_signal = (ratio_tok / ratio_tok.mean(1, keepdim=True)).mean(0)
                if args.peak_smooth > 0:
                    # centredness: average the signal over a box of peak_smooth x the window footprint (in glance
                    # tokens), so a centre needs support around it, not one isolated token
                    side = min(int((args.tile_tokens ** 0.5) * PATCH), max(image.width, image.height))
                    kx = max(1, round(args.peak_smooth * side / image.width * cols))
                    ky = max(1, round(args.peak_smooth * side / image.height * rows))
                    grid = torch.nn.functional.pad(peak_signal.view(1, 1, rows, cols),
                                                   (kx // 2, (kx - 1) // 2, ky // 2, (ky - 1) // 2), mode="replicate")
                    peak_signal = torch.nn.functional.avg_pool2d(grid, (ky, kx), stride=1).flatten()
                t_peak = clock() - start
                for k in args.k:
                    start = clock()
                    chosen_views = peak_views(peak_signal, rows, cols, image.width, image.height, args.tile_tokens, k)
                    lp, ntok, nfill = read([zoom_piece(image, v, args.tile_tokens) for v in chosen_views])
                    t_read = clock() - start
                    row[f"peak-{k}"] = {"correct": int(lp[letter_ids].argmax()) == answer,
                                        "tokens": ntok, "prefill": nfill, "seconds": t_glance + t_peak + t_read,
                                        "read": t_read, "boxes": [v.box for v in chosen_views],
                                        "probs": lp[letter_ids].softmax(0).tolist()}
            if not args.no_downscale:
                for k in args.k:
                    budget = row[f"fovea-{k}"]["tokens"]  # matched to the read (the glance passes are extra)
                    start = clock()
                    lp, prompt, _ = reader.log_probs([Piece(resized(image, budget))], text)
                    row[f"downscale-{k}"] = {"correct": int(lp[letter_ids].argmax()) == answer,
                                             "tokens": prompt.num_visual, "seconds": clock() - start}
            for size in args.vicrop:
                start = clock()
                small = resized(image, size)

                def attention(prompt_text):
                    def make(prompt, spans):
                        last = int(spans[-1].max())
                        return GlanceRecorder(spans[0], torch.arange(last + 1, prompt.length,
                                                                     device=prompt.embeds.device))
                    _, _, rec = reader.log_probs([Piece(small)], prompt_text, make)
                    return rec.per_token()[:, 0].float()
                ratio = (attention(text) / (attention(GENERIC) + 1e-6)).mean(0).cpu()
                owner_v = token_owner(views, small.height // PATCH, small.width // PATCH).float()
                vic = owner_v @ ratio / owner_v.sum(1).clamp_min(1)
                t_vic = clock() - start
                for k in args.k:
                    start = clock()
                    chosen = sorted(pick(vic, boxes_list, k))
                    pieces = [Piece(overview_image, OVERVIEW)] + [zoom_piece(image, views[i], args.tile_tokens)
                                                                for i in chosen]
                    lp, prompt, _ = reader.log_probs(pieces, text)
                    row[f"vicrop{size}-{k}"] = {"correct": int(lp[letter_ids].argmax()) == answer,
                                                "tokens": prompt.num_visual, "seconds": t_vic + clock() - start,
                                                "views": chosen}
            for name, ranking in selectors.items():
                if not args.verify:
                    continue
                # single-view reads of the top max(M) proposals (greedy NMS order, so top-M is a prefix); stored
                # with their answer probabilities so gating policies can be evaluated offline
                singles = []
                for i in pick(ranking, boxes_list, max(args.verify)):
                    start = clock()
                    lp, ntok, nfill = read([zoom_piece(image, views[i], args.tile_tokens)])
                    probs = lp[letter_ids].softmax(0)
                    singles.append({"view": i, "tokens": ntok, "prefill": nfill, "seconds": clock() - start,
                                    "probs": probs.tolist(), "correct": int(probs.argmax()) == answer})
                row[f"{name}-singles"] = singles
                for m in args.verify:
                    entropy = lambda s: -sum(p * torch.tensor(p).clamp_min(1e-9).log().item() for p in s["probs"])  # noqa: E731
                    best = min(singles[:m], key=entropy)
                    row[f"{name}-verify-{m}"] = {"correct": best["correct"],
                                                 "tokens": sum(s["tokens"] for s in singles[:m]),
                                                 "seconds": pres[name] + sum(s["seconds"] for s in singles[:m]),
                                                 "view": best["view"]}
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            torch.cuda.empty_cache()
            print(f"[{number + 1}/{len(samples)}] {sample['id']} ref={row.get('reference', {}).get('seconds', 0):.2f}s "
                  f"pre={pre:.3f}s (glance {t_glance:.3f}, features {t_features:.3f}) "
                  + " ".join(f"{k}={'ok' if v['correct'] else 'x'}" for k, v in row.items()
                             if isinstance(v, dict) and "correct" in v), flush=True)


if __name__ == "__main__":
    main()

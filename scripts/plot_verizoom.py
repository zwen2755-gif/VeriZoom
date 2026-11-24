"""Figures of the VeriZoom paper in the style of the CVPR paper (scripts/plot_cvpr.py); every number is read from the
run outputs in runs/fovea.

    python scripts/plot_verizoom.py numbers     # runs/fovea/paper_numbers.json (accuracy / latency / TFLOPs / tokens)
    python scripts/plot_verizoom.py framework   # Figure 2
    python scripts/plot_verizoom.py teaser      # Figure 1
    python scripts/plot_verizoom.py results     # accuracy against compute on the four benchmarks
    python scripts/plot_verizoom.py efficiency  # latency, TFLOPs, peak memory
    python scripts/plot_verizoom.py analysis    # proposals, verification, exits, read budget
    python scripts/plot_verizoom.py qualitative
Writes paper_verizoom/fig/<name>.pdf (and a .png preview).
"""
import json
import sys
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fovea_cascade_an import cascade  # noqa: E402
from fovea_label import LOADERS, question_text  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
OUT = ROOT / "outputs" / "fig"
RUN = ROOT / "runs" / "fovea"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.weight": "bold", "mathtext.fontset": "dejavusans",
    "mathtext.default": "bf", "font.size": 7.5, "axes.linewidth": 1.4, "axes.edgecolor": "#222222",
    "axes.labelsize": 7.6, "axes.labelweight": "bold", "axes.labelcolor": "#111111", "axes.titleweight": "bold",
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "xtick.color": "#111111", "ytick.color": "#111111",
    "xtick.major.width": 1.2, "ytick.major.width": 1.2, "xtick.major.size": 3.2, "ytick.major.size": 3.2,
    "ytick.minor.size": 1.8, "ytick.minor.width": 0.8, "legend.fontsize": 6.8, "legend.frameon": True,
    "legend.framealpha": 0.96, "legend.edgecolor": "#666666", "lines.linewidth": 1.8, "lines.markersize": 5.0,
    "lines.markeredgecolor": "black", "lines.markeredgewidth": 0.7, "patch.linewidth": 0.8, "pdf.fonttype": 42,
    "hatch.linewidth": 0.45})
OURS, OURS_LIGHT, OURS_MID = "#0F766E", "#D5F0EA", "#5EC4B0"
INK, MUTED = "#222222", "#6b6b6b"
NAME = "VeriZoom"
# one colour, marker and hatch per method in every figure
STYLE = {
    "Downscale": ("#94A3B8", "v", "...."), "Full image": ("#555555", "s", ""), "ViCrop": ("#CA8A04", "^", "////"),
    "AttWarp": ("#9F1239", "D", "\\\\\\\\"), "HiDe": ("#D97706", "P", "xxx"), NAME: (OURS, "o", "")}
SETS = {"V*": ["vstar"], "HR-4K": ["hrbench4k", "hrbench4k_c123"], "HR-8K": ["hrbench8k", "hrbench8k_c123"],
        "MME-RW": ["mmerw"]}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.015, dpi=220)
    print(OUT / f"{name}.pdf")


def header(ax, text, color=INK, y=1.06):
    ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.6, fontweight="bold",
            color=color, bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.35", fc="#f3f4f6", ec="none"))


def logticks(ax, axis="x", choices=(1, 10, 100, 1000)):
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    lo, hi = (ax.get_xlim() if axis == "x" else ax.get_ylim())
    ticks = [t for t in choices if lo <= t <= hi]
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_locator(NullLocator())
    a.set_minor_formatter(NullFormatter())


def tidy(ax, grid_axis="both"):
    ax.grid(axis=grid_axis, color="#999999", lw=0.6, alpha=0.55, which="major")
    ax.set_axisbelow(True)
    ax.tick_params(which="both", direction="out")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)


# ------------------------------------------------------------------------------------------------ numbers
def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def rows_of(name, ds):
    paths = [RUN / d / name for d in ds]
    if not all(p.exists() for p in paths):
        return None
    rows = sum((load(p) for p in paths), [])
    full = {"V*": 191, "HR": 800, "MME": 1919}["V*" if ds[0] == "vstar" else "MME" if ds[0] == "mmerw" else "HR"]
    return rows if len(rows) == full else None


LLM = {"3b": (2048, 36, 256, 11008), "7b": (3584, 28, 512, 18944)}
VD, VP, TXT = 1280, 32 * (4 * 1280 * 1280 + 3 * 1280 * 3420), 80


def flops_fns(model):
    D, L, KV, FF = LLM[model]
    P = L * (2 * D * D + 2 * D * KV + 3 * D * FF)

    def llm(tq, ctx=None):
        ctx = ctx or tq
        return 2 * P * tq + 4 * D * L * tq * (ctx - tq / 2)

    def dec(steps, ctx):
        return steps * (2 * P + 4 * D * L * ctx)
    return llm, dec


def vit(merged):
    n = 4 * merged
    return 2 * VP * n + 4 * VD * n * (28 * 64 + 4 * n)


def config(model):
    return json.load(open(RUN / "mmerwtrain" / ("cascade_h_config.json" if model == "3b" else "cascade_7b_config.json")))


def ours_rows(rows, model, cand):
    """per question: correct, seconds, flops, tokens encoded, exit stage"""
    llm, _ = flops_fns(model)
    c = config(model)
    out = []
    for r in rows:
        a, s, e = cascade([r], c["rule"], c["refine"], c["Tg"], c["Tf"], c["tau_g"], c["tau_f"])
        st = e.index(100)
        g = r["grid"][0] * r["grid"][1]
        cut = g + 20
        f, t = vit(g) + llm(g + TXT), g
        if st >= 1:
            k = len(r["boxes"])
            f += k * (vit(cand) + llm(cand + TXT, cut + cand + TXT))
            t += k * cand
        if st == 2:
            rd = r["read"][c["rule"]]
            v = rd[c["refine"]]["visual"] - g if rd[c["refine"]].get("visual") else sum(rd["tokens"])
            f += vit(v) + llm(v + TXT, cut + v + TXT)
            t += v
        out.append((a > 50, s, f, t, st))
    return out


def summarize(per):
    n = len(per)
    return {"acc": 100 * sum(p[0] for p in per) / n, "sec": sum(p[1] for p in per) / n,
            "tflops": sum(p[2] for p in per) / n / 1e12, "tokens": sum(p[3] for p in per) / n, "n": n,
            "correct": [bool(p[0]) for p in per]}


def numbers():
    res = {}
    for name, ds in SETS.items():
        R = {}
        for model, tag, cand, key in (("3b", "test_h", 256, f"{NAME} (3B)"), ("3b", "b1536_576", 576, f"{NAME}-1536 (3B)"),
                                      ("3b", "b3072_576", 576, f"{NAME}-3072 (3B)"), ("7b", "test_7b", 256, f"{NAME} (7B)")):
            rows = rows_of(f"cascade_{tag}.jsonl", ds)
            if rows:
                R[key] = summarize(ours_rows(rows, model, cand))
        for model in ("3b", "7b"):
            llm, dec = flops_fns(model)
            for tag in (f"native_{model}.jsonl", f"native_{model}1600.jsonl"):
                rows = rows_of(tag, ds)
                if not rows:
                    continue
                for t in ("1600", "8192"):
                    if t not in rows[0]:
                        continue
                    lab = f"{'Downscale' if t == '1600' else 'Full image'} ({model.upper()})"
                    R[lab] = summarize([(r[t]["correct"], r[t]["seconds"], vit(r[t]["tokens"]) + llm(r[t]["tokens"] + TXT),
                                         r[t]["tokens"], 2) for r in rows])
        llm, dec = flops_fns("3b")
        rows = rows_of("hide_3b.jsonl", ds)
        if rows:
            R["HiDe (3B)"] = summarize([(r["correct"], r["seconds"], llm(300) + dec(60, 360) + vit(r["visual_tokens_attention"])
                                         + llm(r["visual_tokens_attention"] + TXT) + vit(r["visual_tokens_attention"])
                                         + vit(r["visual_tokens"] - r["visual_tokens_attention"]) + llm(r["visual_tokens"] + TXT),
                                         r["visual_tokens_attention"] + r["visual_tokens"], 2) for r in rows])
        rows = rows_of("attwarp_3b.jsonl", ds)
        if rows:
            R["AttWarp (3B)"] = summarize([(r["correct"], r["seconds"], vit(r["visual_tokens_attention"])
                                            + llm(r["visual_tokens_attention"] + TXT)
                                            + dec(r["attention_steps"], r["visual_tokens_attention"] + TXT)
                                            + vit(r["visual_tokens"]) + llm(r["visual_tokens"] + TXT),
                                            r["visual_tokens_attention"] + r["visual_tokens"], 2) for r in rows])
        rows = rows_of("e2e_fast.jsonl", ds)
        if rows and "vicrop-1" in rows[0]:
            per = []
            for r in rows:
                v, g = r["vicrop-1"], r["glance"]["tokens"]
                # two attention passes over the 1,024-token image (question and generic prompt), then one read of
                # a 256-token overview and the selected 576-token crop
                f = vit(g) + 2 * llm(g + TXT) + vit(v["tokens"]) + llm(v["tokens"] + TXT)
                per.append((v["correct"], v["seconds"], f, g + v["tokens"], 2))
            R["ViCrop (3B)"] = summarize(per)
        # 7B rows measured on the RTX PRO 6000: accuracy and TFLOPs are reported, latency is not comparable
        llm7, _ = flops_fns("7b")
        rows = rows_pro("native_7b8192_pro.jsonl", ds)
        if rows:
            R["Full image (7B)"] = summarize([(r["8192"]["correct"], 0.0, vit(r["8192"]["tokens"])
                                               + llm7(r["8192"]["tokens"] + TXT), r["8192"]["tokens"], 2) for r in rows])
            R["Full image (7B)"]["sec"] = None
        rows = rows_pro("e2e_fast7b_pro.jsonl", ds)
        if rows and "vicrop-1" in rows[0]:
            per = []
            for r in rows:
                v, g = r["vicrop-1"], r["glance"]["tokens"]
                per.append((v["correct"], 0.0, vit(g) + 2 * llm7(g + TXT) + vit(v["tokens"]) + llm7(v["tokens"] + TXT),
                            g + v["tokens"], 2))
            R["ViCrop (7B)"] = summarize(per)
            R["ViCrop (7B)"]["sec"] = None
        res[name] = R
    json.dump(res, open(RUN / "paper_numbers.json", "w"), indent=1)
    for name, R in res.items():
        print(name)
        for k, v in R.items():
            print(f"  {k:22s} {v['acc']:5.1f}  {v['sec'] or 0:6.2f}s  {v['tflops']:7.1f} TF  {v['tokens']:7.0f} tok  n={v['n']}")
    return res


def get_numbers():
    p = RUN / "paper_numbers.json"
    return json.load(open(p)) if p.exists() else numbers()


def mcnemar(a, b):
    x = sum(1 for u, v in zip(a, b) if u and not v)
    y = sum(1 for u, v in zip(a, b) if v and not u)
    n = x + y
    return min(1, 2 * sum(comb(n, k) for k in range(min(x, y) + 1)) / 2 ** n) if n else 1.0


# ------------------------------------------------------------------------------------------------ framework
EX = "direct_attributes/sa_25835"
LET = ["black", "green", "yellow", "red"]        # options A-D of the example as shown to the model
BOXC = ["#94A3B8", "#D97706", OURS, "#4338CA"]


def example():
    s = [s for s in LOADERS["vstar"][0](ROOT / LOADERS["vstar"][1]) if s["id"] == EX][0]
    r = [x for x in load(RUN / "vstar" / "cascade_test_h.jsonl") if x["id"] == EX][0]
    return s, r, Image.open(s["image"]).convert("RGB")


def panel_box(fig, x, y, w, h, FW, FH, num, title):
    fig.add_artist(patches.FancyBboxPatch((x / FW, y / FH), w / FW, h / FH, boxstyle="round,pad=0,rounding_size=0.012",
                                          fc="#F8FAFC", ec="#CBD5E1", lw=1.0, transform=fig.transFigure, zorder=-10))
    fig.text((x + 0.1) / FW, (y + h - 0.16) / FH, str(num), fontsize=8.5, color="white", ha="center", va="center",
             fontweight="bold", bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.2", fc="#334155", ec="none"))
    fig.text((x + w / 2 + 0.08) / FW, (y + h - 0.16) / FH, title, fontsize=8.6, color=INK, ha="center", va="center",
             fontweight="bold")


def framework():
    s, r, img = example()
    W, H = img.size
    small = img.resize((W // 4, H // 4))
    w, chosen, toks = r["read"]["both"]["w"], r["read"]["both"]["chosen"], r["read"]["both"]["tokens"]
    FW, FH = 7.0, 2.6
    fig = plt.figure(figsize=(FW, FH))
    A = lambda x, y, wi, hi: fig.add_axes([x / FW, y / FH, wi / FW, hi / FH])
    T = lambda x, y, t, **kw: fig.text(x / FW, y / FH, t, **kw)

    def frame(ax, color, lw=1.4):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(color); sp.set_linewidth(lw)

    def bars(ax, p, title):
        hi = int(np.argmax(p))
        ax.barh(range(4), p, color=["#D97706" if i == hi else "#CBD5E1" for i in range(4)], height=0.7,
                edgecolor="black", linewidth=0.4)
        ax.set_xlim(0, 1.25); ax.set_yticks(range(4), LET, fontsize=5.6); ax.invert_yaxis(); ax.set_xticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0, pad=1.5)
        for i, v in enumerate(p):
            ax.text(v + 0.04, i, f"{v:.2f}", va="center", fontsize=5.3, color=INK if i == hi else MUTED)
        ax.set_title(title, fontsize=6.2, color=MUTED, pad=2)
        ax.set_facecolor("none")

    def arrow(p0, p1, color=OURS, ls="-", rad=0.0):
        fig.patches.append(patches.FancyArrowPatch((p0[0] / FW, p0[1] / FH), (p1[0] / FW, p1[1] / FH),
                                                   transform=fig.transFigure, arrowstyle="-|>", mutation_scale=8,
                                                   lw=1.2, color=color, linestyle=ls, shrinkA=0, shrinkB=0,
                                                   connectionstyle=f"arc3,rad={rad}"))

    # panel 1: glance and proposals
    panel_box(fig, 0.0, 0.0, 2.2, FH, FW, FH, 1, "Glance and propose")
    iw = 1.45
    ih = iw * H / W
    ax = A(0.08, 1.12, iw, ih)
    ax.imshow(small)
    pi_path = RUN / "figure" / "pi_sa_25835.pt"
    if pi_path.exists():
        pi = torch.load(pi_path).float().numpy()
        ax.imshow(pi / pi.max(), extent=(0, small.width, small.height, 0), cmap="inferno", alpha=0.5,
                  interpolation="bilinear")
    for k, b in enumerate(r["boxes"]):
        ax.add_patch(patches.Rectangle((b[0] / 4, b[1] / 4), (b[2] - b[0]) / 4, (b[3] - b[1]) / 4, fill=False,
                                       ec=BOXC[k], lw=1.3))
        lx, ly, ha, va = (b[2] / 4 + 4, b[1] / 4 + 2, "left", "top") if k != 3 else (b[0] / 4 - 4, b[3] / 4, "right", "bottom")
        ax.text(lx, ly, f"{k + 1}", color="white", fontsize=5.6, va=va, ha=ha, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.12", fc=BOXC[k], ec="none"))
    frame(ax, "#94A3B8", 0.8)
    T(0.08 + iw + 0.08, 1.12 + ih - 0.05, "1,024\ntokens", fontsize=6.0, color=MUTED, va="top")
    T(0.08 + iw + 0.08, 1.12 + 0.3, "cache\n$G$", fontsize=6.4, color=OURS, va="top")
    T(1.1, 0.98, r"$\pi$: question-word attention" "\n" "of 3 localization heads", fontsize=6.0, color=MUTED,
      ha="center", va="top")
    bars(A(0.55, 0.12, 1.1, 0.42), r["p0"], r"glance answer $p_0$")

    # panel 2: verification
    x2 = 2.28
    panel_box(fig, x2, 0.0, 2.72, FH, FW, FH, 2, "Verify on forks of the glance cache")
    t = 0.5
    for k, b in enumerate(r["boxes"]):
        cx = x2 + 0.12 + k * 0.645
        a = A(cx, 1.42, t, t)
        a.imshow(img.crop(b).resize((200, 200)))
        frame(a, BOXC[k], 2.0 if k in chosen else 1.0)
        T(cx + 0.02, 1.42 + t - 0.02, f"{k + 1}", color="white", fontsize=5.6, va="top", fontweight="bold",
          bbox=dict(boxstyle="square,pad=0.12", fc=BOXC[k], ec="none"))
        p = r["P1"][k]
        j = int(np.argmax(p))
        T(cx, 1.27, f"{LET[j]} {p[j]:.2f}", fontsize=5.9, color=INK)
        T(cx, 1.12, f"$s_{k + 1}$ {r['share'][k]:.2f}", fontsize=5.9, color=MUTED)
        T(cx, 0.96, f"$w_{k + 1}$ {w[k]:.2f}", fontsize=6.4, color=OURS if k in chosen else MUTED)
        arrow((cx + t / 2, 2.12), (cx + t / 2, 1.93), color="#94A3B8")
    T(x2 + 1.36, 2.17, r"$[\tilde I,\ I[b_k],\ q]$ on a fork of $G$, 256 tokens per region", fontsize=5.9,
      color=MUTED, ha="center")
    fig.patches.append(patches.FancyBboxPatch(((x2 + 0.32) / FW, 0.18 / FH), 2.08 / FW, 0.56 / FH,
                                              boxstyle="round,pad=0,rounding_size=0.012", fc="white", ec=OURS, lw=1.2,
                                              transform=fig.transFigure))
    T(x2 + 1.36, 0.53, r"$w_k\propto s_k\cdot e^{H(p_0)-H(P_k)}$", fontsize=7.6, color=OURS, ha="center", va="center")
    T(x2 + 1.36, 0.3, "attention share × confidence gain", fontsize=6.0, color=MUTED, ha="center", va="center")
    arrow((2.2 - 0.06, 1.75), (x2 + 0.08, 1.75))

    # panel 3: read and exit
    x3 = 5.08
    panel_box(fig, x3, 0.0, FW - x3, FH, FW, FH, 3, "Read and exit")
    big = 0.74
    for i, k in sorted(enumerate(chosen), key=lambda z: -toks[z[0]]):
        side = big * (toks[i] / max(toks)) ** 0.5
        x = x3 + 0.12 if toks[i] == max(toks) else x3 + 0.12 + big + 0.1
        a = A(x, 1.3, side, side)
        a.imshow(img.crop(r["boxes"][k]))
        frame(a, BOXC[k], 2.0)
        T(x, 1.17, f"{toks[i]} tok", fontsize=5.8, color=MUTED)
    T(x3 + 0.12, 2.17, r"verified regions, $N_k\propto w_k$", fontsize=5.9, color=MUTED)
    bars(A(x3 + 0.5, 0.12, 0.9, 0.42), r["read"]["both"]["r0"]["probs"], r"read answer $p_2$")
    T(x3 + 0.95, 0.88, "exit at the first stage with\n" r"$m(p_0)\geq\tau_g$  or  $m(p_1)\geq\tau_f$", fontsize=5.8,
      color=MUTED, ha="center", va="center")
    arrow((x2 + 2.72 - 0.06, 1.75), (x3 + 0.08, 1.75))
    save(fig, "framework")


# ------------------------------------------------------------------------------------------------ teaser
def scatter_panel(ax, R, title, xlabel="TFLOPs per question (log)", budgets=True):
    order = [("Downscale (3B)", "Downscale"), ("ViCrop (3B)", "ViCrop"), ("Full image (3B)", "Full image"),
             ("AttWarp (3B)", "AttWarp"), ("HiDe (3B)", "HiDe")]
    for key, lab in order:
        if key in R:
            c, m, _ = STYLE[lab]
            ax.scatter(R[key]["tflops"], R[key]["acc"], s=46, color=c, marker=m, edgecolor="black", lw=0.7, zorder=4,
                       label=lab)
    pts = [R[k] for k in (f"{NAME} (3B)", f"{NAME}-1536 (3B)", f"{NAME}-3072 (3B)") if k in R]
    ax.plot([p["tflops"] for p in pts], [p["acc"] for p in pts], "-o", color=OURS, lw=2.2, ms=6.5, mec="black",
            mew=0.7, zorder=5, label=NAME)
    for i, (p, b) in enumerate(zip(pts, ("576", "1536", "3072") if budgets else ())):
        up = p["acc"] >= max(q["acc"] for q in pts) - 1e-9
        off = {0: (0, -11), 1: (-4, 6) if up else (0, -11), 2: (12, -10) if not up else (10, 5)}[i]
        ax.annotate(b, (p["tflops"], p["acc"]), textcoords="offset points", xytext=off, ha="center",
                    fontsize=5.4, color=OURS)
    ax.set_xscale("log")
    lo = min(min(v["tflops"] for v in R.values()), 15) / 1.3
    ax.set_xlim(lo, max(v["tflops"] for v in R.values()) * 1.5)
    logticks(ax, choices=(20, 50, 200, 500))
    ax.set_xlabel(xlabel, labelpad=1.5)
    tidy(ax)
    header(ax, title)


def teaser():
    """Figure 1. Left: an HR-Bench 8K question that every compared method misses except VeriZoom, with each method's
    answer and seconds (all from the run logs). Right: accuracy against seconds per question on HR-Bench 8K, marker
    area proportional to TFLOPs."""
    import random
    from matplotlib.patches import ConnectionPatch
    ds, qid = "hrbench8k", "single/284"
    c = config("3b")
    s = [x for x in LOADERS[ds][0](ROOT / LOADERS[ds][1]) if x["id"] == qid][0]
    r = [x for x in load(RUN / ds / "cascade_test_h.jsonl") if x["id"] == qid][0]
    order = list(range(len(s["options"])))
    if not s.get("keep_order"):
        random.Random(s["id"]).shuffle(order)
    opt = [s["options"][order[i]] for i in range(len(order))]
    pred, _ = final_answer(r, c)
    nat = [x for x in load(RUN / ds / "native_3b.jsonl") if x["id"] == qid][0]["8192"]
    hide = [x for x in load(RUN / ds / "hide_3b.jsonl") if x["id"] == qid][0]
    att = [x for x in load(RUN / ds / "attwarp_3b.jsonl") if x["id"] == qid][0]
    vic = [x for x in load(RUN / ds / "e2e_fast.jsonl") if x["id"] == qid][0]["vicrop-1"]
    ours_sec = (r["seconds"]["glance"] + r["seconds"]["propose"] + r["seconds"]["focus"]
                + r["read"]["both"]["r0"]["seconds"])
    rows = [("Glance", int(np.argmax(r["p0"])), r["seconds"]["glance"]),
            ("Full image", int(np.argmax(nat["probs"])), nat["seconds"]),
            ("ViCrop", int(np.argmax(vic["probs"])), vic["seconds"]),
            ("AttWarp", att["pred"], att["seconds"]),
            ("HiDe", hide["pred"], hide["seconds"]),
            (NAME, pred, ours_sec)]
    img = Image.open(s["image"]).convert("RGB")
    W, H = img.size
    rd = r["read"]["both"]
    k = rd["chosen"][int(np.argmax([rd["w"][i] for i in rd["chosen"]]))]
    b = r["boxes"][k]
    area = 100 * (b[2] - b[0]) * (b[3] - b[1]) / (W * H)

    FW, FH = 7.0, 2.8
    fig = plt.figure(figsize=(FW, FH))
    A = lambda x, y, w, h: fig.add_axes([x / FW, y / FH, w / FW, h / FH])
    # full image with the verified region
    iw = 1.85
    ih = iw * H / W
    ax = A(0.0, FH - 0.24 - ih, iw, ih)
    ax.imshow(img.resize((1100, int(1100 * H / W))), extent=(0, W, H, 0))
    ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=OURS, lw=1.8))
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#475569"); sp.set_linewidth(1.0)
    fig.text(0.0, (FH - 0.2) / FH, f"HR-Bench 8K  |  {W:,} \u00d7 {H:,} pixels", fontsize=7.4, color="white",
             va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.25", fc="#334155", ec="none"))
    ax.text(b[0] - 120, b[1] - 120, f"{area:.1f}% of\nthe image", color="white", fontsize=6.2, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc=OURS, ec="none"))
    # the verified region, enlarged
    side = ih
    az = A(iw + 0.12, FH - 0.24 - ih, side, side)
    az.imshow(img.crop(b))
    az.set_xticks([]); az.set_yticks([])
    for sp in az.spines.values():
        sp.set_color(OURS); sp.set_linewidth(2.2)
    az.text(0.03, 0.97, "verified region", transform=az.transAxes, fontsize=6.2, color="white", va="top",
            bbox=dict(boxstyle="square,pad=0.15", fc=OURS, ec="none"))
    for (px, py), (qx, qy) in (((b[2], b[1]), (0, 1)), ((b[2], b[3]), (0, 0))):
        fig.add_artist(ConnectionPatch((px, py), (qx, qy), coordsA="data", coordsB="axes fraction", axesA=ax,
                                       axesB=az, color=OURS, lw=1.6, ls=(0, (3, 1.8))))
    fig.text(0.0, (FH - 0.3 - ih) / FH, "Q: " + s["question"], fontsize=7.0, color=INK, va="top", ha="left",
             fontweight="bold")
    # answers and seconds of every method on this question
    axb = A(0.62, 0.3, iw + side + 0.12 - 0.62 + 0.25, FH - 0.24 - ih - 0.62)
    for i, (lab, a, sec) in enumerate(rows):
        ok = a == r["answer"]
        colr = OURS if lab == NAME else ("#555555" if lab in ("Glance", "Full image") else STYLE[lab][0])
        axb.barh(i, sec, color=colr, edgecolor="black", lw=0.5, height=0.7, hatch=STYLE.get(lab, ("", "", ""))[2],
                 zorder=3)
        mark = "\u2713" if ok else "\u2717"
        txt = f"{opt[a]}  {mark}   {sec:.1f} s"
        inside = sec > 8
        axb.text(sec - 0.3 if inside else sec + 0.3, i, txt, va="center", ha="right" if inside else "left",
                 fontsize=6.0, color=OURS if ok else "#B91C1C", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9) if inside else None,
                 zorder=4)
    axb.set_yticks(range(len(rows)), [x[0] for x in rows], fontsize=6.2)
    axb.invert_yaxis()
    axb.set_xlim(0, 21)
    axb.set_xlabel("Seconds for this question", fontsize=6.4, labelpad=1)
    tidy(axb, "x")
    # right: accuracy against seconds, marker area ~ TFLOPs
    N = get_numbers()["HR-8K"]
    ax2 = A(4.35, 0.42, 2.6, FH - 0.72)
    pts = [("Downscale (3B)", "Downscale"), ("ViCrop (3B)", "ViCrop"), ("Full image (3B)", "Full image"),
           ("AttWarp (3B)", "AttWarp"), ("HiDe (3B)", "HiDe")]
    size = lambda tf: 3.0 * tf ** 0.85  # noqa: E731
    for key, lab in pts:
        v = N[key]
        col, m, _ = STYLE[lab]
        ax2.scatter(v["sec"], v["acc"], s=size(v["tflops"]), color=col, marker="o", edgecolor="black", lw=0.7,
                    alpha=0.9, zorder=4)
        dx, dy, ha = {"Downscale": (1.25, 0, "left"), "ViCrop": (1.25, -0.2, "left"), "Full image": (1.25, 0.9, "left"),
                      "AttWarp": (1.0, -3.2, "center"), "HiDe": (1.0, -3.1, "center")}[lab]
        ax2.text(v["sec"] * dx, v["acc"] + dy, lab, fontsize=6.4, color=col if lab != "Full image" else "#333333",
                 ha=ha, va="center", fontweight="bold")
    ours = [N[f"{NAME} (3B)"], N[f"{NAME}-1536 (3B)"], N[f"{NAME}-3072 (3B)"]]
    ax2.plot([p["sec"] for p in ours], [p["acc"] for p in ours], "-", color=OURS, lw=2.0, zorder=3)
    for p, bud in zip(ours, (576, 1536, 3072)):
        ax2.scatter(p["sec"], p["acc"], s=size(p["tflops"]), color=OURS, edgecolor="black", lw=0.7, zorder=5)
    ax2.text(ours[0]["sec"] * 0.93, ours[0]["acc"] - 1.6, NAME, fontsize=7.0, color=OURS, ha="center",
             fontweight="bold")
    h, o = N["HiDe (3B)"], ours[1]
    ax2.annotate("", xy=(o["sec"] * 1.25, o["acc"] + 0.4), xytext=(h["sec"] * 0.75, h["acc"] + 0.2),
                 arrowprops=dict(arrowstyle="-|>", color=OURS, lw=1.8, connectionstyle="arc3,rad=0.25"), zorder=6)
    tf_ratio = int(h["tflops"] / o["tflops"])
    sec_ratio = int(h["sec"] / o["sec"])
    ax2.text(4.6, 71.2, f"{sec_ratio}\u00d7 faster,  {tf_ratio}\u00d7 fewer TFLOPs\n"
             f"{h['acc'] - o['acc']:.1f} points from HiDe", fontsize=6.4, color=OURS, ha="center", va="center",
             fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=OURS, lw=1.0))
    ax2.set_xscale("log")
    ax2.set_xlim(0.25, 60)
    ax2.set_ylim(49, 73.5)
    logticks(ax2, choices=(0.3, 1, 3, 10, 30))
    ax2.set_xlabel("Seconds per question (log)", labelpad=1.5)
    ax2.set_ylabel("Accuracy (%)", labelpad=1.5)
    tidy(ax2)
    header(ax2, "HR-Bench 8K, Qwen2.5-VL-3B", y=1.02)
    ax2.text(0.98, 0.03, "marker area: TFLOPs", transform=ax2.transAxes, fontsize=5.6, color=MUTED, ha="right")
    print("example", qid, "area", round(area, 2), [(l, opt[a], round(t, 2)) for l, a, t in rows])
    print("ratios", sec_ratio, tf_ratio, h["sec"] / o["sec"], h["tflops"] / o["tflops"], h["acc"] - o["acc"])
    save(fig, "teaser")


# ------------------------------------------------------------------------------------------------ results
def results():
    N = get_numbers()
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9), gridspec_kw={"wspace": 0.42})
    for ax, (name, title) in zip(axes, (("V*", "(a) V*"), ("HR-4K", "(b) HR-Bench 4K"), ("HR-8K", "(c) HR-Bench 8K"),
                                        ("MME-RW", "(d) MME-RealWorld"))):
        scatter_panel(ax, N[name], title, xlabel="TFLOPs (log)", budgets=False)
    axes[0].set_ylabel("Accuracy (%)", labelpad=1.5)
    h, l = [], []
    for a in axes:
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    l = [x if x != NAME else f"{NAME} (B = 576, 1536, 3072)" for x in l]
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=6, fontsize=6.2, handlelength=1.4,
               columnspacing=1.0)
    save(fig, "results")


# ------------------------------------------------------------------------------------------------ efficiency
def efficiency():
    """(a) latency and (b) TFLOPs per question of each benchmark, one full-width row each so that every value reads
    horizontally; (c) peak memory on HR-Bench 8K spans both rows. One legend row on top."""
    N = get_numbers()
    fig = plt.figure(figsize=(7.0, 3.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.0, 1.2], hspace=0.55, wspace=0.2)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])]
    methods = [("Full image (3B)", "Full image"), ("AttWarp (3B)", "AttWarp"), ("HiDe (3B)", "HiDe"),
               (f"{NAME}-1536 (3B)", f"{NAME}, B=1536"), (f"{NAME} (3B)", f"{NAME}, B=576")]
    sets = ["V*", "HR-4K", "HR-8K", "MME-RW"]
    names = {"V*": "V*", "HR-4K": "HR-Bench 4K", "HR-8K": "HR-Bench 8K", "MME-RW": "MME-RealWorld"}
    fmt = lambda v: f"{v:.1f}" if v < 10 else f"{v:.0f}"  # noqa: E731
    for ax, key, yl, title, lim, ticks in (
            (axes[0], "sec", "Seconds (log)", "(a) Latency per question", (0.3, 60), (1, 10)),
            (axes[1], "tflops", "TFLOPs (log)", "(b) Compute per question", (12, 2500), (10, 100, 1000))):
        wdt = 0.165
        for j, (m, lab) in enumerate(methods):
            if lab.startswith(NAME):
                c, hatch = (OURS_MID, "///") if "1536" in lab else (OURS, "")
            else:
                c, _, hatch = STYLE[lab]
            xs = np.arange(len(sets)) + (j - 2) * wdt
            vals = [N[s][m][key] for s in sets]
            ax.bar(xs, vals, width=wdt * 0.9, color=c, edgecolor="black", linewidth=0.5, hatch=hatch, label=lab,
                   zorder=3)
            for x, v in zip(xs, vals):
                ours = lab == f"{NAME}, B=576"
                ax.text(x, v * 1.1, fmt(v), ha="center", va="bottom", fontsize=6.0 if key == "sec" else 5.6, zorder=4,
                        fontweight="bold" if ours else "normal", color="#0B5D52" if ours else INK)
        ax.set_yscale("log")
        ax.set_ylim(*lim)
        logticks(ax, "y", choices=ticks)
        ax.set_xticks(range(len(sets)), [names[s] for s in sets], fontsize=6.8)
        ax.set_xlim(-0.55, len(sets) - 0.45)
        ax.set_ylabel(yl, labelpad=1.5)
        tidy(ax, "y")
        header(ax, title)
    mem = {}
    for line in open(ROOT / "logs" / "sampler" / "queue28.log"):
        p = line.split()
        if len(p) == 4 and p[0] == "hrbench8k" and p[2] == "PEAK_GB":
            mem[p[1]] = float(p[3])
    ax = axes[2]
    rows = [("AttWarp", "attwarp"), ("HiDe", "hide"), (NAME, "ours")]
    for i, (lab, k) in enumerate(rows):
        c, _, hatch = STYLE[lab]
        ax.bar(i, mem[k], width=0.6, color=c, edgecolor="black", linewidth=0.5, hatch=hatch, zorder=3)
        ax.text(i, mem[k] + 0.25, f"{mem[k]:.1f}", ha="center", va="bottom", fontsize=7.0, zorder=4,
                fontweight="bold" if k == "ours" else "normal", color="#0B5D52" if k == "ours" else INK)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=6.6)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, 15)
    ax.set_ylabel("Peak GPU memory (GB)", labelpad=1.5)
    tidy(ax, "y")
    header(ax, "(c) Memory, HR-Bench 8K")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.985), ncol=5, fontsize=7.0, handlelength=1.6,
               columnspacing=1.4, frameon=False)
    save(fig, "efficiency")


# ------------------------------------------------------------------------------------------------ analysis (1x4)
def analysis():
    N = get_numbers()
    sets = ["V*", "HR-4K", "HR-8K", "MME-RW"]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95), gridspec_kw={"wspace": 0.55})
    pos = axes[3].get_position()
    axes[3].set_position([pos.x0 + 0.05, pos.y0, pos.width, pos.height])
    # (a) gain over the glance of reading one candidate: the highest mode of pi (light) and, on top, the extra gain
    # of reading the candidate with the highest weight after verification (dark); 3B solid, 7B hatched
    ax = axes[0]
    stroke = [matplotlib.patheffects.withStroke(linewidth=1.8, foreground="white")]
    for mi, (m, tag, hatch) in enumerate((("3B", "test_h", ""), ("7B", "test_7b", "////"))):
        for si, st in enumerate(sets):
            R = rows_of(f"cascade_{tag}.jsonl", SETS[st])
            g = 100 * np.mean([int(np.argmax(r["p0"])) == r["answer"] for r in R])
            top = 100 * np.mean([int(np.argmax(r["P1"][0])) == r["answer"] for r in R])
            kw = [int(np.argmax(r["read"]["both"]["w"])) for r in R]
            ver = 100 * np.mean([int(np.argmax(r["P1"][k])) == r["answer"] for r, k in zip(R, kw)])
            print(m, st, "glance", round(g, 1), "top of pi", round(top, 1), "top of w", round(ver, 1))
            x = si + (mi - 0.5) * 0.38
            a, b = top - g, ver - top
            ax.bar(x, a, width=0.36, color="#99D5CC", edgecolor="black", lw=0.6, hatch=hatch, zorder=3,
                   label="Highest mode of $\\pi$" if mi == 0 and si == 0 else None)
            ax.bar(x, b, bottom=a, width=0.36, color=OURS, edgecolor="black", lw=0.6, hatch=hatch, zorder=3,
                   label="+ Verification" if mi == 0 and si == 0 else None)
            ax.text(x, max(a, a + b) + 0.35, f"{a + b:.0f}", ha="center", va="bottom", fontsize=5.2,
                    color="#0B5D52", fontweight="bold", path_effects=stroke, zorder=6)
    ax.bar([0], [0], color="white", edgecolor="black", lw=0.6, label="3B")
    ax.bar([0], [0], color="white", edgecolor="black", lw=0.6, hatch="////", label="7B")
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xticks(range(4), ["V*", "4K", "8K", "MME"], fontsize=5.8)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-2, 17)
    ax.set_ylabel("Gain over glance (points)", labelpad=1.5)
    tidy(ax, "y")
    header(ax, "(a) Region choice")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=5.2, handlelength=1.3,
              columnspacing=0.6, labelspacing=0.2, handletextpad=0.3)
    # (b) read budget
    ax = axes[1]
    cols = {"V*": ("#4338CA", "D"), "HR-4K": ("#D97706", "s"), "HR-8K": ("#9F1239", "^"), "MME-RW": (OURS, "o")}
    keys = [f"{NAME} (3B)", f"{NAME}-1536 (3B)", f"{NAME}-3072 (3B)"]
    for st in sets:
        c, m = cols[st]
        ys = [N[st][k]["acc"] - N[st][keys[0]]["acc"] for k in keys]
        ax.plot([576, 1536, 3072], ys, "-", marker=m, color=c, lw=1.8, ms=4.8, mec="black", mew=0.6, label=st, zorder=3)
    ax.axhline(0, color="#555555", lw=0.8, ls=(0, (3, 2)))
    ax.set_xscale("log")
    ax.set_xticks([576, 1536, 3072], ["576", "1536", "3072"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Read budget $B$", labelpad=1.5)
    ax.set_ylabel(r"$\Delta$ accuracy vs. $B=576$", labelpad=1.5)
    tidy(ax)
    header(ax, "(b) Read budget, 3B")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, fontsize=5.6, handlelength=1.3,
              columnspacing=0.6, labelspacing=0.2)
    # (c) glance threshold sweep, mean over the four benchmarks: accuracy (solid, left) and latency (dashed, right)
    ax = axes[2]
    ax2 = ax.twinx()
    grid = [round(0.05 * k, 2) for k in range(21)]
    handles = []
    for model, c, m in (("3b", OURS, "o"), ("7b", "#4338CA", "s")):
        cf = config(model)
        tag = "test_h" if model == "3b" else "test_7b"
        rows = {st: rows_of(f"cascade_{tag}.jsonl", ds) for st, ds in SETS.items()}
        xs, ys = [], []
        for tg in grid:
            a = [cascade(rows[st], cf["rule"], cf["refine"], cf["Tg"], cf["Tf"], tg, cf["tau_f"]) for st in sets]
            xs.append(sum(x[1] for x in a) / 4); ys.append(sum(x[0] for x in a) / 4)
        mk = list(range(0, len(grid), 2))
        h, = ax.plot(grid, ys, "-", marker=m, markevery=mk, color=c, lw=1.7, ms=3.6, mec="black", mew=0.5,
                     label=model.upper(), zorder=4)
        ax2.plot(grid, xs, "--", color=c, lw=1.3, alpha=0.85, zorder=3)
        handles.append(h)
        sel = [cascade(rows[st], cf["rule"], cf["refine"], cf["Tg"], cf["Tf"], cf["tau_g"], cf["tau_f"]) for st in sets]
        sy = sum(x[0] for x in sel) / 4
        hs = ax.scatter(cf["tau_g"], sy, s=95, marker="*", color="#F59E0B", edgecolor="black", lw=0.6, zorder=6,
                        label="Selected $\\tau_g$")
        print(model, "tau_g", cf["tau_g"], "acc", round(sy, 2), "sec", round(sum(x[1] for x in sel) / 4, 3),
              "glance only", round(ys[0], 1), round(xs[0], 2), "tau 1", round(ys[-1], 1), round(xs[-1], 2))
    handles += [matplotlib.lines.Line2D([], [], color="#333333", lw=1.5, label="Accuracy"),
                matplotlib.lines.Line2D([], [], color="#333333", lw=1.3, ls="--", label="Latency"), hs]
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(54.5, 67.0)
    ax2.set_ylim(0.0, 1.45)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel(r"Glance threshold $\tau_g$", labelpad=1.5)
    ax.set_ylabel("Mean accuracy (%)", labelpad=1.5)
    ax2.set_ylabel("Seconds (dashed)", labelpad=2)
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.tick_params(axis="y", pad=1.5)
    for label in ax2.get_yticklabels():
        label.set_fontweight("bold")
    tidy(ax)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    header(ax, r"(c) Glance exit $\tau_g$")
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3, fontsize=5.4,
              handlelength=1.5, columnspacing=0.6, labelspacing=0.2, handletextpad=0.3)
    # (d) exits: share of questions leaving at each stage (bars) and their accuracy (diamonds), mean over the four
    # benchmarks, both in percent on one axis
    ax = axes[3]
    stages_lab = ["Glance", "Verify", "Read"]
    bar_h, acc_h = [], []
    for mi, (model, tag, colr, dark, hatch) in enumerate((("3b", "test_h", OURS, "#0B5D52", ""),
                                                         ("7b", "test_7b", "#A5B4FC", "#3730A3", "///"))):
        cf = config(model)
        share, acc = np.zeros(3), np.zeros(3)
        for st in sets:
            rows = rows_of(f"cascade_{tag}.jsonl", SETS[st])
            fa = [final_answer(r, cf) for r in rows]
            for k in range(3):
                idx = [n for n, (_, sk) in enumerate(fa) if sk == k]
                share[k] += 100 * len(idx) / len(rows) / len(sets)
                acc[k] += 100 * sum(fa[n][0] == rows[n]["answer"] for n in idx) / max(1, len(idx)) / len(sets)
        print(model, "exit share", share.round(1), "exit accuracy", acc.round(1))
        xs = np.arange(3) + (mi - 0.5) * 0.36
        bar_h.append(ax.bar(xs, share, width=0.34, color=colr, edgecolor="black", lw=0.7, hatch=hatch,
                            label=f"{model.upper()} share", zorder=3))
        acc_h.append(ax.plot(xs, acc, "D", ms=4.6, mfc=dark, mec="black", mew=0.6, label=f"{model.upper()} acc.",
                             zorder=5)[0])
        for x, v, sh in zip(xs, acc, share):
            below = 0 < (v + 4.5) - sh < 7 or abs(v + 6 - sh) < 4     # keep the number off the top edge of the bar
            ax.text(x, v - 4.5 if below else v + 4.5, f"{v:.0f}", ha="center", va="top" if below else "bottom",
                    fontsize=5.4, color=dark, fontweight="bold", zorder=6,
                    path_effects=[matplotlib.patheffects.withStroke(linewidth=1.8, foreground="white")])
    ax.set_xticks(range(3), stages_lab, fontsize=5.8)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Questions / accuracy (%)", labelpad=1.5)
    tidy(ax, "y")
    header(ax, "(d) Exit stage")
    ax.legend(handles=bar_h + acc_h, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=5.2,
              handlelength=1.2, columnspacing=0.4, labelspacing=0.2, handletextpad=0.3)
    save(fig, "analysis")


# ------------------------------------------------------------------------------------------------ tables
def fmt_col(vals, higher=True, marks=True):
    """best / second marks for one column of a block"""
    ok = sorted({round(v, 1) for v in vals if v is not None}, reverse=higher)
    out = []
    for v in vals:
        if v is None:
            out.append("--")
            continue
        t = f"{v:.1f}" if abs(v) < 100 else f"{v:.0f}"
        if marks and round(v, 1) == ok[0]:
            t = r"\best{" + t + "}"
        elif marks and len(ok) > 1 and round(v, 1) == ok[1]:
            t = r"\second{" + t + "}"
        out.append(t)
    return out


def lat7b_seconds():
    """{row key: {set: mean seconds}} of the Qwen2.5-VL-7B rows from the A800 latency runs (GPU to itself, same
    questions for every method: every 2nd of V* / HR-Bench, every 19th of MME-RealWorld)"""
    c = config("7b")
    out = {}
    for st, ds in SETS.items():
        d = ds[0]
        rc, rn, re_ = (load(RUNP / d / f"{k}_lat_7b.jsonl") for k in ("cascade", "native", "e2e"))
        assert len(rc) == len(rn) == len(re_) and [r["id"] for r in rc] == [r["id"] for r in rn], d
        out.setdefault(f"{NAME} (7B)", {})[st] = cascade(rc, c["rule"], c["refine"], c["Tg"], c["Tf"], c["tau_g"],
                                                       c["tau_f"])[1]
        out.setdefault("Downscale (7B)", {})[st] = float(np.mean([r["1600"]["seconds"] for r in rn]))
        out.setdefault("Full image (7B)", {})[st] = float(np.mean([r["8192"]["seconds"] for r in rn]))
        out.setdefault("ViCrop (7B)", {})[st] = float(np.mean([r["vicrop-1"]["seconds"] for r in re_]))
    for k, v in out.items():
        print("A800 latency", k, {st: round(x, 2) for st, x in v.items()})
    return out


def tables():
    N = get_numbers()
    sets = ["V*", "HR-4K", "HR-8K", "MME-RW"]
    blocks = [("Qwen2.5-VL-3B", [("Downscale (1,600 tokens)", "Downscale (3B)", ""),
                                 ("Full image (8,192 tokens)", "Full image (3B)", ""),
                                 (r"ViCrop~\cite{zhang2025vicrop}", "ViCrop (3B)", ""),
                                 (r"AttWarp~\cite{dalal2026attwarp}", "AttWarp (3B)", ""),
                                 (r"HiDe~\cite{liu2026hide}", "HiDe (3B)", ""),
                                 (r"\name{} ($B=576$)", f"{NAME} (3B)", "ours"),
                                 (r"\name{} ($B=1536$)", f"{NAME}-1536 (3B)", "ours")]),
              ("Qwen2.5-VL-7B", [("Downscale (1,600 tokens)", "Downscale (7B)", ""),
                                 ("Full image (8,192 tokens)", "Full image (7B)", ""),
                                 (r"ViCrop~\cite{zhang2025vicrop}", "ViCrop (7B)", ""),
                                 (r"\name{} ($B=576$)", f"{NAME} (7B)", "ours")])]
    lines = []
    for title, rows in blocks:
        cols = [[N[st][k]["acc"] if k in N[st] else None for _, k, _ in rows] for st in sets]
        full = [all(k in N[st] for st in sets) for _, k, _ in rows]
        avg = [sum(c[i] for c in cols) / 4 if full[i] else None for i in range(len(rows))]
        tf = [sum(N[st][k]["tflops"] for st in sets) / 4 if full[i] else None for i, (_, k, _) in enumerate(rows)]
        if title.endswith("7B"):          # 7B: dedicated A800 latency runs of all four rows (server/lat_7b_a800.sh)
            L7 = lat7b_seconds()
            sec = [sum(L7[k][st] for st in sets) / 4 for _, k, _ in rows]
        else:
            sec = [sum(N[st][k]["sec"] for st in sets) / 4
                   if full[i] and all(N[st][k]["sec"] is not None for st in sets) else None
                   for i, (_, k, _) in enumerate(rows)]
        F = [fmt_col(c) for c in cols] + [fmt_col(avg), fmt_col(tf, False, False), fmt_col(sec, False, False)]
        lines.append(r"\multicolumn{8}{l}{\textit{" + title + r"}}\\")
        for i, (lab, k, tag) in enumerate(rows):
            if tag == "ours" and rows[i - 1][2] != "ours":
                lines.append(r"\midrule")
            lines.append((r"\rowcolor{oursbg} " if tag == "ours" else "") + lab + " & "
                         + " & ".join(f[i] for f in F) + r" \\")
        lines.append(r"\midrule")
    body = "\n".join("    " + l for l in lines[:-1])
    tex = (r"""\begin{table*}[t]
  \centering
  \caption{Accuracy (\%) on four high-resolution benchmarks and mean cost per question. HR-Bench is averaged over its four option rotations. TFLOPs and latency are averaged over the four benchmarks. Latency is measured on one RTX 5080 GPU for 3B and on one A800 GPU for 7B, on the same questions for all methods of a model, every second question of V* and HR-Bench and every 19th of MME-RealWorld for 7B.}
  \label{tab:main}
  \small
  \setlength{\tabcolsep}{8pt}
  \begin{tabular}{l cccc c cc}
    \toprule
    Method & V* & HR-Bench 4K & HR-Bench 8K & MME-RW & Avg. & TFLOPs & Latency (s) \\
    \midrule
""" + body + r"""
    \bottomrule
  \end{tabular}
\end{table*}
""")
    (ROOT / "outputs" / "tab").mkdir(exist_ok=True)
    (ROOT / "outputs" / "tab" / "main.tex").write_text(tex, encoding="utf-8")
    print(tex)


def stage_accuracy(model):
    """{set: {variant: (acc, seconds)}} from the stored stages of the test runs"""
    tag = "test_h" if model == "3b" else "test_7b"
    c = config(model)
    am = lambda p: int(torch.tensor(p).argmax())
    out = {}
    for st, ds in SETS.items():
        rows = rows_of(f"cascade_{tag}.jsonl", ds)
        n = len(rows)
        g = sum(r["seconds"]["glance"] for r in rows) / n
        f = g + sum(r["seconds"]["propose"] + r["seconds"]["focus"] for r in rows) / n
        acc = lambda pred: 100 * sum(pred(r) == r["answer"] for r in rows) / n
        rd = lambda rule, form: sum(r["read"][rule][form]["seconds"] for r in rows) / n
        v = {"glance": (acc(lambda r: am(r["p0"])), g),
             "p1_uniform": (acc(lambda r: am(r["read"]["uniform"]["p1"])), f),
             "p1_both": (acc(lambda r: am(r["read"]["both"]["p1"])), f)}
        for rule in ("uniform", "both"):
            for form in ("r0", "w0", "s0"):
                v[f"{rule}_{form}"] = (acc(lambda r: am(r["read"][rule][form]["probs"])), f + rd(rule, form))
        cas = [cascade([r], c["rule"], c["refine"], c["Tg"], c["Tf"], c["tau_g"], c["tau_f"]) for r in rows]
        v["cascade"] = (sum(x[0] for x in cas) / n, sum(x[1] for x in cas) / n)
        out[st] = v
    return out


def ablation_tables():
    sets = ["V*", "HR-4K", "HR-8K", "MME-RW"]
    S = {m: stage_accuracy(m) for m in ("3b", "7b")}
    lines = []
    for m, title in (("3b", "Qwen2.5-VL-3B"), ("7b", "Qwen2.5-VL-7B")):
        form = config(m)["refine"]
        rows = [("Glance only", "glance"), ("+ Verify, uniform weights", "p1_uniform"),
                (r"+ Verify, weights of \Cref{eq:weights}", "p1_both"),
                ("+ Read, uniform weights", f"uniform_{form}"), (r"+ Read, weights of \Cref{eq:weights}", f"both_{form}"),
                ("+ Early exits (full model)", "cascade")]
        cols = [[S[m][st][k][0] for _, k in rows] for st in sets]
        avg = [sum(c[i] for c in cols) / 4 for i in range(len(rows))]
        sec = [sum(S[m][st][k][1] for st in sets) / 4 for _, k in rows]
        F = [fmt_col(c) for c in cols] + [fmt_col(avg), fmt_col(sec, False, False)]
        lines.append(r"\multicolumn{7}{l}{\textit{" + title + r"}}\\")
        for i, (lab, k) in enumerate(rows):
            lines.append((r"\rowcolor{oursbg} " if k == "cascade" else "") + lab + " & " + " & ".join(f[i] for f in F)
                         + r" \\")
        lines.append(r"\midrule")
    body = "\n".join("    " + l for l in lines[:-1])
    tex = (r"""\begin{table}[t]
  \centering
  \caption{Ablation of the stages of \name{} with $B=576$. Accuracy (\%) and mean latency per question.}
  \label{tab:ablation}
  \footnotesize
  \setlength{\tabcolsep}{2.4pt}
  \begin{tabular}{l cccc c c}
    \toprule
    Variant & V* & 4K & 8K & MME & Avg. & Sec. \\
    \midrule
""" + body + r"""
    \bottomrule
  \end{tabular}
\end{table}
""")
    (ROOT / "outputs" / "tab" / "ablation.tex").write_text(tex, encoding="utf-8")
    print(tex)
    # read forms, always read with the weights of Eq. (weights)
    lines = []
    for m, title in (("3b", "3B"), ("7b", "7B")):
        rows = [("Separate crops", "both_r0"), ("Warped, after the glance", "both_w0"), ("Warped, alone", "both_s0")]
        cols = [[S[m][st][k][0] for _, k in rows] for st in sets]
        avg = [sum(c[i] for c in cols) / 4 for i in range(len(rows))]
        F = [fmt_col(c) for c in cols] + [fmt_col(avg)]
        sel = {"r0": 0, "w0": 1, "s0": 2}[config(m)["refine"]]
        for i, (lab, k) in enumerate(rows):
            lines.append((r"\multirow{3}{*}{" + title + "}" if i == 0 else "") + " & " + lab
                         + (r"$^\dagger$" if i == sel else "") + " & " + " & ".join(f[i] for f in F) + r" \\")
        lines.append(r"\midrule")
    body = "\n".join("    " + l for l in lines[:-1])
    tex = (r"""\begin{table}[t]
  \centering
  \caption{Read forms of \name{} without early exits. $^\dagger$ marks the form selected on the training split.}
  \label{tab:forms}
  \footnotesize
  \setlength{\tabcolsep}{3pt}
  \begin{tabular}{ll cccc c}
    \toprule
    & Read form & V* & 4K & 8K & MME & Avg. \\
    \midrule
""" + body + r"""
    \bottomrule
  \end{tabular}
\end{table}
""")
    (ROOT / "outputs" / "tab" / "forms.tex").write_text(tex, encoding="utf-8")
    print(tex)


# ------------------------------------------------------------------------------------------------ qualitative
QUAL = [("V*", "vstar"), ("HR-Bench 4K", "hrbench4k"), ("HR-Bench 8K", "hrbench8k"), ("MME-RealWorld", "mmerw")]


def final_answer(r, c):
    acc, _, e = cascade([r], c["rule"], c["refine"], c["Tg"], c["Tf"], c["tau_g"], c["tau_f"])
    st = e.index(100)
    if st == 0:
        p = r["p0"]
    elif st == 1:
        p = r["read"][c["rule"]]["p1"]
    else:
        p = r["read"][c["rule"]][c["refine"]]["probs"]
    return int(np.argmax(p)), st


def short_opt(o):
    o = str(o).strip()
    if len(o) > 26 and " is " in o:          # V* options are sentences: "The color of the bag is red."
        o = o.rsplit(" is ", 1)[1]
    return o.rstrip(".")


def qual_candidates(ds, c, prefix="", maxopt=26, aspect=(1.2, 2.0)):
    """questions the glance gets wrong and VeriZoom (3B) gets right, smallest verified region first"""
    import random
    samples = {s["id"]: s for s in LOADERS[ds][0](ROOT / LOADERS[ds][1])}
    hide = {x["id"]: x for x in load(RUN / ds / "hide_3b.jsonl")}
    out, seen = [], set()
    for r in load(RUN / ds / "cascade_test_h.jsonl"):
        s = samples.get(r["id"])
        if s is None or r["id"] not in hide or r["id"] in seen or not r["id"].startswith(prefix):
            continue
        seen.add(r["id"])
        pred, st = final_answer(r, c)
        if pred != r["answer"] or int(np.argmax(r["p0"])) == r["answer"] or st == 0:
            continue
        _, gold, letters = question_text(s)
        if letters[r["answer"]] != gold:
            continue
        order = list(range(len(s["options"])))
        if not s.get("keep_order"):
            random.Random(s["id"]).shuffle(order)
        shown = {r["answer"], int(np.argmax(r["p0"])), int(hide[r["id"]]["pred"])}
        if len(s["question"]) > 110 or max(len(short_opt(s["options"][order[i]])) for i in shown) > maxopt:
            continue
        if r["id"] == EX:      # already shown in Figures 1 and 2
            continue
        W, H = Image.open(s["image"]).size
        if not aspect[0] <= W / H <= aspect[1]:
            continue
        rd = r["read"][c["rule"]]
        k = rd["chosen"][int(np.argmax([rd["w"][i] for i in rd["chosen"]]))]
        b = r["boxes"][k]
        if "bboxes" in s and s["bboxes"]:
            cx = [(x + w / 2, y + h / 2) for x, y, w, h in s["bboxes"]]
            if not any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for x, y in cx):
                continue
        area = (b[2] - b[0]) * (b[3] - b[1]) / (W * H)
        out.append((area, s, r, k, pred, st, hide[r["id"]]))
    return sorted(out, key=lambda z: z[0])


# (tab title, dataset, id prefix, index into the candidate list)
QUAL_SETS = [("V*", "vstar", "", "direct_attributes/sa_33492"), ("HR-Bench 4K", "hrbench4k", "", "cross/636"),
             ("HR-Bench 8K", "hrbench8k", "", "single/228"),
             ("MME-RealWorld", "mmerw", "", "perception/ocr_cc/license/h1_0018")]
QUAL_TASKS = [("V* | Relative position", "vstar", "relative_position", "relative_position/sa_34785", 80),
              ("HR-8K | Cross-instance", "hrbench8k", "cross", "cross/560"),
              ("MME | Remote sensing", "mmerw", "perception/remote_sensing", "perception/remote_sensing/position/3376"),
              ("MME | Driving", "mmerw", "Perception/Autonomous_Driving/Attribute_Motion_Vehicle", 0),
              ("MME | Monitoring", "mmerw", "perception/monitoring", "perception/monitoring/vehicle/counting/0138"),
              ("MME | Diagram and table", "mmerw", "perception/diagram_and_table",
               "perception/diagram_and_table/diagram/4090"),
              ("MME | Product text", "mmerw", "perception/ocr_cc/adver_and_product",
               "perception/ocr_cc/adver_and_product/0164"),
              ("MME | Reasoning", "mmerw", "reason", "reasoning/diagram_and_table/diagram/0074")]

# eight more task types, none of them shown in QUAL_SETS or QUAL_TASKS
QUAL_MORE = [("V* | Object category", "vstar", "direct_attributes", "direct_attributes/sa_47397"),
             ("HR-8K | Map reading", "hrbench8k", "cross", "cross/616"),
             ("HR-8K | Brand knowledge", "hrbench8k", "single", "single/748"),
             ("HR-8K | Arrow direction", "hrbench8k", "single", "single/284"),
             ("MME | Sign meaning", "mmerw", "reasoning/ocr_cc/scene", "reasoning/ocr_cc/scene/0011"),
             ("MME | Ego-vehicle action", "mmerw", "Reasoning/Autonomous_Driving/Relation_Interaction_Ego2Vehicle",
              "Reasoning/Autonomous_Driving/Relation_Interaction_Ego2Vehicle/0005"),
             ("MME | Intention", "mmerw", "reason/monitoring/intention", "reason/monitoring/intention/0333"),
             ("MME | Material", "mmerw", "reason/monitoring/property", "reason/monitoring/property/0453"),
             ("MME | Occupation", "mmerw", "reasoning/ocr_cc/char_id", "reasoning/ocr_cc/char_id/0175"),
             ("MME | Price", "mmerw", "perception/ocr_cc/text_recog", "perception/ocr_cc/text_recog/0693"),
             ("MME | Table lookup", "mmerw", "perception/diagram_and_table/table",
              "perception/diagram_and_table/table/2631"),
             ("MME | Fax number", "mmerw", "perception/ocr_cc/phone_and_address",
              "perception/ocr_cc/phone_and_address/0055")]


def qual_panel(fig, FW, FH, x0, ybase, uw, ph, spec, c):
    import random
    import textwrap
    from matplotlib.patches import ConnectionPatch
    A = lambda x, y, w, h: fig.add_axes([x / FW, y / FH, w / FW, h / FH])
    T = lambda x, y, t, **kw: fig.text(x / FW, y / FH, t, **kw)
    title, ds, prefix, idx = spec[:4]
    maxopt = spec[4] if len(spec) > 4 else 60
    cand = qual_candidates(ds, c, prefix, maxopt=maxopt, aspect=(0.75, 5.0))
    print(title, len(cand), [z[1]["id"] for z in cand[:6]])
    if isinstance(idx, str):
        cand = [z for z in cand if z[1]["id"] == idx]
    area, s, r, k, pred, st, hd = cand[0] if isinstance(idx, str) else cand[min(idx, len(cand) - 1)]
    print("   ->", s["id"], f"area {100 * area:.1f}%", "exit stage", st)
    _, _, letters = question_text(s)
    order = list(range(len(s["options"])))
    if not s.get("keep_order"):
        random.Random(s["id"]).shuffle(order)
    opt = lambda i: f"{letters[i]}. {short_opt(s['options'][order[i]])}"
    img = Image.open(s["image"]).convert("RGB")
    W, H = img.size
    fig.add_artist(patches.FancyBboxPatch(((x0 - 0.03) / FW, (ybase + 0.02) / FH), (uw + 0.06) / FW, (ph - 0.04) / FH,
                                          boxstyle="round,pad=0,rounding_size=0.012", fc="none", ec="#475569",
                                          lw=1.8, transform=fig.transFigure, zorder=-10))
    top = ybase + ph - 0.02
    T(x0 + uw / 2, top - 0.11, title, fontsize=7.2, color="white", ha="center", va="center", fontweight="bold",
      bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.3", fc="#334155", ec="none"))
    a = W / H
    wide = a > 2.4          # panoramas: image across the panel, enlarged region below it next to the question
    q = textwrap.fill(s["question"], 19 if wide else 36)
    nq = q.count("\n") + 1
    text_h = 0.07 + 0.1 * nq + 0.05 + 3 * 0.115
    ytop = top - 0.25
    if wide:
        h = uw / a
        side = 0.62
        ax = A(x0, ytop - h, uw, h)
    else:
        h = min(0.82, (uw - 0.06) / (a + 0.62), ph - 0.29 - text_h - 0.06)
        side = 0.62 * h
        ax = A(x0, ytop - h, h * a, h)
    ax.imshow(img.resize((900, max(1, int(900 / a)))), extent=(0, W, H, 0))
    rd = r["read"][c["rule"]]
    for j, b in enumerate(r["boxes"]):
        sel = j in rd["chosen"]
        ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                                       ec=OURS if j == k else ("#D97706" if sel else "white"),
                                       lw=1.6 if sel else 0.8, ls="-" if sel else (0, (2, 1.5))))
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#475569"); sp.set_linewidth(1.2)
    b = r["boxes"][k]
    crop = img.crop(b)
    cw, ch = crop.size
    m = max(cw, ch)
    sq = Image.new("RGB", (m, m), (255, 255, 255))
    sq.paste(crop, ((m - cw) // 2, (m - ch) // 2))
    az = A(x0 + uw - side, ytop - h - 0.08 - side, side, side) if wide else A(x0 + uw - side, ytop - h, side, side)
    az.imshow(sq)
    az.set_xticks([]); az.set_yticks([])
    for sp in az.spines.values():
        sp.set_color(OURS); sp.set_linewidth(2.2)
    links = (((b[0], b[3]), (0, 1)), ((b[2], b[3]), (1, 1))) if wide else (((b[2], b[1]), (0, 1)), ((b[2], b[3]), (0, 0)))
    for (px, py), (qx, qy) in links:
        fig.add_artist(ConnectionPatch((px, py), (qx, qy), coordsA="data", coordsB="axes fraction", axesA=ax,
                                       axesB=az, color=OURS, lw=1.6, ls=(0, (3, 1.8))))
    ty = ytop - h - 0.07
    T(x0, ty, q, fontsize=5.6, color=INK, va="top", fontweight="bold", linespacing=1.15)
    ty = ybase + 0.06 + 3 * 0.115
    lines = [("Glance", int(np.argmax(r["p0"])), "#555555"), ("HiDe", int(hd["pred"]), STYLE["HiDe"][0]),
             (NAME, pred, OURS)]
    for lab, ans, colr in lines:
        ok = ans == r["answer"]
        T(x0, ty, f"{lab}:", fontsize=5.6, color=colr, va="top", fontweight="bold")
        T(x0 + 0.5, ty, textwrap.shorten(opt(ans), 22, placeholder="...") + ("  ✓" if ok else "  ✗"),
          fontsize=5.6, color=INK if ok else "#B91C1C", va="top", fontweight="bold")
        ty -= 0.115


def qual_figure(specs, name, ph=1.75):
    c = config("3b")
    FW, CW = 7.0, 7.0 / 4
    nrow = (len(specs) + 3) // 4
    FH = ph * nrow
    fig = plt.figure(figsize=(FW, FH))
    for n, spec in enumerate(specs):
        row, col = divmod(n, 4)
        qual_panel(fig, FW, FH, col * CW + 0.05, FH - (row + 1) * ph, CW - 0.1, ph, spec, c)
    save(fig, name)


def qualitative():
    qual_figure(QUAL_SETS, "qualitative")


def qualitative_tasks():
    qual_figure(QUAL_TASKS, "qualitative_tasks")



def qualitative_more():
    qual_figure(QUAL_MORE, "qualitative_more")

# ------------------------------------------------------------------------------------------------ task types
TASKS = [("V*", "Attribute", "vstar", lambda i: i.startswith("direct_attributes")),
         ("V*", "Spatial", "vstar", lambda i: i.startswith("relative_position")),
         ("HR-4K", "Single", "HR-4K", lambda i: i.startswith("single")),
         ("HR-4K", "Cross", "HR-4K", lambda i: i.startswith("cross")),
         ("HR-8K", "Single", "HR-8K", lambda i: i.startswith("single")),
         ("HR-8K", "Cross", "HR-8K", lambda i: i.startswith("cross")),
         ("MME", "OCR", "mmerw", lambda i: "ocr_cc" in i),
         ("MME", "Remote sensing", "mmerw", lambda i: "remote_sensing" in i),
         ("MME", "Diagram, table", "mmerw", lambda i: "diagram_and_table" in i),
         ("MME", "Monitoring", "mmerw", lambda i: "monitoring" in i),
         ("MME", "Driving", "mmerw", lambda i: "Autonomous_Driving" in i)]
TASK_SETS = {"vstar": ["vstar"], "HR-4K": SETS["HR-4K"], "HR-8K": SETS["HR-8K"], "mmerw": ["mmerw"]}


def per_row_correct(key):
    """{set: [(id, correct)]} for one method, 3B"""
    out = {}
    for name, ds in TASK_SETS.items():
        if key == NAME:
            c = config("3b")
            rows = rows_of("cascade_test_h.jsonl", ds)
            out[name] = [(r["id"], final_answer(r, c)[0] == r["answer"]) for r in rows]
        elif key in ("Downscale", "Full image"):
            t = "1600" if key == "Downscale" else "8192"
            for tag in ("native_3b.jsonl", "native_3b1600.jsonl"):
                rows = rows_of(tag, ds)
                if rows and t in rows[0]:
                    out[name] = [(r["id"], bool(r[t]["correct"])) for r in rows]
        elif key == "ViCrop":
            out[name] = [(r["id"], bool(r["vicrop-1"]["correct"])) for r in rows_of("e2e_fast.jsonl", ds)]
        else:
            f = {"AttWarp": "attwarp_3b.jsonl", "HiDe": "hide_3b.jsonl"}[key]
            out[name] = [(r["id"], bool(r["correct"])) for r in rows_of(f, ds)]
    return out


def task_table():
    methods = ["Downscale", "Full image", "ViCrop", "AttWarp", "HiDe", NAME]
    per = {m: per_row_correct(m) for m in methods}
    rows = []
    for group, lab, st, sel in TASKS:
        accs, n = [], 0
        for m in methods:
            xs = [c for i, c in per[m][st] if sel(i)]
            n = len(xs)
            accs.append(100 * sum(xs) / n)
        rows.append((group, lab, n, accs))
        print(f"{group:6s} {lab:16s} n={n:4d} " + " ".join(f"{a:5.1f}" for a in accs))
    head = " & ".join(["Downscale", "Full", "ViCrop", "AttWarp", "HiDe", r"\name{}"])
    body, last = [], None
    for group, lab, n, accs in rows:
        if last is not None and group != last:
            body.append(r"    \midrule")
        cells = fmt_col(accs)
        body.append(f"    {group if group != last else ''} & {lab} & " + " & ".join(cells[:-1])
                    + r" & \cellcolor{oursbg}" + cells[-1] + r" \\")
        last = group
    tex = (r"""\begin{table}[t]
  \centering
  \caption{Accuracy (\%) by question type with Qwen2.5-VL-3B. Full is the image at 8,192 tokens.}
  \label{tab:tasks}
  \footnotesize
  \setlength{\tabcolsep}{2.6pt}
  \begin{tabular}{ll cccccc}
    \toprule
     & Type & """ + head + r""" \\
    \midrule
""" + "\n".join(body) + r"""
    \bottomrule
  \end{tabular}
\end{table}
""")
    (ROOT / "outputs" / "tab" / "tasks.tex").write_text(tex, encoding="utf-8")
    print(tex)


# ------------------------------------------------------------------------------------------------ candidates
def verified_answer(r, K):
    """p1 recomputed from the stored verification passes of the first K candidates (modes of pi in order)"""
    Pk = np.array(r["P1"][:K])
    w = np.array(r["share"][:K]) * np.exp(r["H0"] - np.array(r["Hk"][:K]))
    return int(((w / w.sum())[:, None] * Pk).sum(0).argmax())


def k_table():
    """compact: mean accuracy of p1 over the four benchmarks against K, and the verification TFLOPs"""
    lines = []
    for model, tag in (("3b", "test_h"), ("7b", "test_7b")):
        llm, _ = flops_fns(model)
        acc, tf = [], []
        for K in (1, 2, 3, 4):
            a, t = [], []
            for st, ds in SETS.items():
                rows = rows_of(f"cascade_{tag}.jsonl", ds)
                a.append(100 * np.mean([verified_answer(r, K) == r["answer"] for r in rows]))
                t.append(np.mean([K * (vit(256) + llm(256 + TXT, r["grid"][0] * r["grid"][1] + 20 + 256 + TXT))
                                  for r in rows]) / 1e12)
            acc.append(float(np.mean(a))); tf.append(float(np.mean(t)))
        print(model, [round(x, 1) for x in acc], [round(x, 1) for x in tf])
        lines.append(f"    {model.upper()}, accuracy & " + " & ".join(fmt_col(acc)) + r" \\")
        lines.append(f"    {model.upper()}, TFLOPs & " + " & ".join(f"{x:.1f}" for x in tf) + r" \\")
    tex = (r"""\begin{table}[t]
  \centering
  \caption{Verified answer $p_1$ with the first $K$ modes of $\pi$ as candidates. Mean accuracy (\%) over the four benchmarks and TFLOPs of the verification passes.}
  \label{tab:candidates}
  \footnotesize
  \setlength{\tabcolsep}{6pt}
  \begin{tabular}{l cccc}
    \toprule
     & $K=1$ & $K=2$ & $K=3$ & \cellcolor{oursbg}$K=4$ \\
    \midrule
""" + "\n".join([lines[0], lines[1], r"    \midrule", lines[2], lines[3]]) + r"""
    \bottomrule
  \end{tabular}
\end{table}
""")
    (ROOT / "outputs" / "tab" / "candidates.tex").write_text(tex, encoding="utf-8")
    print(tex)

# ------------------------------------------------------------------------------------------------ inference traces
TRACE = [("Exits after the glance", "V*", "vstar", "direct_attributes/sa_10204"),
         ("Exits after verification", "HR-Bench 8K", "hrbench8k", "single/492"),
         ("Needs the zoom", "MME-RealWorld", "mmerw", "perception/ocr_cc/license/0390")]


def trace():
    """VeriZoom (3B) on one question per exit stage: attention and candidates of the glance, the four candidates with
    their verification weight, the zoomed regions, then the answer next to direct zooming (ViCrop) and the full image.
    Under every stage: its calibrated confidence against its exit threshold (numbers), its answer and its own tokens
    and seconds. Stages that are not run are faded. All seconds from the RTX 5080 runs of the main table."""
    import random
    import textwrap
    from fovea_cascade_an import margin, temp
    c = config("3b")
    FW, RH, TOP = 7.16, 1.66, 0.28
    FH = RH * len(TRACE) + TOP
    fig = plt.figure(figsize=(FW, FH))
    A = lambda x, y, w, h: fig.add_axes([x / FW, y / FH, w / FW, h / FH])
    T = lambda x, y, t, **kw: fig.text(x / FW, y / FH, t, **kw)
    BAD, FADE, GOOD = "#B91C1C", "#E5E7EB", OURS
    PX = 28
    COLS = [(0.06, 1.72, "Glance"), (2.0, 2.02, "Verify"), (4.24, 0.98, "Zoom in"), (5.42, 1.7, "Answer and cost")]
    for n, (x, w, t) in enumerate(COLS):
        T(x + w / 2, FH - 0.12, t, ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
        if n < 3:
            nx, nw = COLS[n + 1][0], COLS[n + 1][1]
            half = 0.075 * len(COLS[n + 1][2]) / 2 + 0.1
            fig.add_artist(patches.FancyArrowPatch(((x + w / 2 + 0.075 * len(t) / 2 + 0.1) / FW, (FH - 0.12) / FH),
                                                   ((nx + nw / 2 - half) / FW, (FH - 0.12) / FH),
                                                   transform=fig.transFigure,
                                                   arrowstyle="-|>,head_width=3.2,head_length=5.5", color="black",
                                                   lw=2.2))

    def ntok(w, h, budget):
        """tokens of a crop resized to about `budget` tokens (fovea_diag_readform.to_tokens)"""
        sc = (budget / ((w / PX) * (h / PX))) ** 0.5
        a, b = max(PX, round(w * sc / PX) * PX), max(PX, round(h * sc / PX) * PX)
        while (a // PX) * (b // PX) > budget:
            a, b = (a - PX, b) if a >= b else (a, b - PX)
        return (a // PX) * (b // PX)

    def stage_info(x, w, y, conf, tau, ans, ok, tok, sec, state):
        """confidence strip with its numbers, then the stage answer and its own cost; state: stop / go / final / off"""
        bw = w - 0.62
        if state == "off":
            return
        fig.add_artist(patches.Rectangle((x / FW, y / FH), bw / FW, 0.07 / FH, transform=fig.transFigure, fc=FADE,
                                         ec="none"))
        col = GOOD if state in ("stop", "final") else "#94A3B8"
        fig.add_artist(patches.Rectangle((x / FW, y / FH), bw * min(1, conf) / FW, 0.07 / FH,
                                         transform=fig.transFigure, fc=col, ec="none"))
        if tau is not None:
            fig.add_artist(Line2D([(x + bw * tau) / FW] * 2, [(y - 0.035) / FH, (y + 0.105) / FH],
                                  transform=fig.transFigure, color=INK, lw=1.4))
            txt = f"{conf:.2f} {'≥' if conf >= tau else '<'} {tau:.2f}"
        else:
            txt = f"{conf:.2f}"
        T(x + bw + 0.06, y + 0.035, txt, fontsize=7.2, fontweight="bold", va="center", color=INK)
        T(x, y - 0.16, f"{ans} {'✓' if ok else '✗'}", fontsize=7.4, fontweight="bold", va="center",
          color=GOOD if ok else BAD)
        if state != "final":
            T(x + w, y - 0.16, "stop" if state == "stop" else "continue", fontsize=7.0, fontweight="bold",
              va="center", ha="right", color=GOOD if state == "stop" else MUTED)
        plus = "" if x < 0.5 else "+"
        T(x, y - 0.32, f"{plus}{tok:,} tokens, {plus}{sec:.2f} s", fontsize=6.8, fontweight="bold", va="center",
          color=MUTED)

    def skipped_box(x, y, w, h, why):
        fig.add_artist(patches.FancyBboxPatch((x / FW, y / FH), w / FW, h / FH,
                                              boxstyle="round,pad=0,rounding_size=0.01", fc="none", ec="#94A3B8",
                                              lw=1.0, ls=(0, (3, 2)), transform=fig.transFigure))
        T(x + w / 2, y + h / 2, why, fontsize=7.2, fontweight="bold", ha="center", va="center", color="#64748B",
          linespacing=1.3)

    for row, (label, setname, ds, qid) in enumerate(TRACE):
        s = [x for x in LOADERS[ds][0](ROOT / LOADERS[ds][1]) if x["id"] == qid][0]
        r = [x for x in load(RUN / ds / "cascade_test_h.jsonl") if x["id"] == qid][0]
        vic = [x for x in load(RUN / ds / "e2e_fast.jsonl") if x["id"] == qid][0]
        nat = [x for x in load(RUN / ds / "native_3b.jsonl") if x["id"] == qid][0]["8192"]
        order = list(range(len(s["options"])))
        if not s.get("keep_order"):
            random.Random(s["id"]).shuffle(order)
        opt = lambda i, n=13: textwrap.shorten(short_opt(s["options"][order[i]]), n, placeholder="...")  # noqa: E731
        pred, stage = final_answer(r, c)
        img = Image.open(s["image"]).convert("RGB")
        W, H = img.size
        rd = r["read"]["both"]
        y0 = FH - TOP - (row + 1) * RH
        if row:
            fig.add_artist(Line2D([0.01, 0.99], [(y0 + RH) / FH] * 2, transform=fig.transFigure, color="#CBD5E1",
                                  lw=0.8))
        T(0.08, y0 + RH - 0.12, setname, fontsize=7.8, fontweight="bold", color=OURS, va="center")
        T(0.08 + 0.09 * len(setname) + 0.12, y0 + RH - 0.12, textwrap.shorten(s["question"], 80, placeholder="..."),
          fontsize=7.8, color=INK, va="center", fontweight="bold")
        # stage answers, confidences and own costs
        p0, p1 = temp(r["p0"], c["Tg"]), temp(rd["p1"], c["Tf"])
        p2 = rd[c["refine"]]["probs"]
        a0, a1, a2 = int(np.argmax(r["p0"])), int(np.argmax(rd["p1"])), int(np.argmax(p2))
        t0 = r["grid"][0] * r["grid"][1]
        t1 = sum(ntok(b[2] - b[0], b[3] - b[1], 256) for b in r["boxes"])
        t2 = sum(rd["tokens"])
        s0, s1 = r["seconds"]["glance"], r["seconds"]["propose"] + r["seconds"]["focus"]
        s2 = rd[c["refine"]]["seconds"]
        img_top, img_bot = y0 + RH - 0.24, y0 + 0.62
        sy = y0 + 0.44
        # (1) glance: image, attention and the four candidates
        ih = img_top - img_bot
        iw = min(1.72, ih * W / H)
        ih = iw * H / W
        ax = A(0.06 + (1.72 - iw) / 2, img_bot + (img_top - img_bot - ih) / 2, iw, ih)
        small = np.asarray(img.resize((700, int(700 * H / W)))).astype(float) / 255
        d = torch.load(RUN / "figure" / f"pi_{ds}_{qid.replace('/', '__')}.pt")    # scripts/fovea_dump_pi.py
        assert d["boxes"] == r["boxes"], (qid, d["boxes"], r["boxes"])           # same attention as the reported run
        sp = torch.nn.functional.interpolate(d["pi"].float()[None, None], size=small.shape[:2], mode="bilinear",
                                             align_corners=False)[0, 0].numpy()
        sp = np.clip(sp / np.percentile(sp, 97), 0, 1) ** 0.6
        ax.imshow(small * (0.55 + 0.45 * sp[..., None]), extent=(0, W, H, 0))
        rgba = plt.get_cmap("jet")(sp)
        rgba[..., 3] = 0.45 * sp
        ax.imshow(rgba, extent=(0, W, H, 0), interpolation="bilinear")
        for k, b in enumerate(r["boxes"]):
            ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=BOXC[k], lw=1.5))
            ax.text(b[0], b[1], f"{k + 1}", color="white", fontsize=6.4, va="top", ha="left", fontweight="bold",
                    bbox=dict(boxstyle="square,pad=0.08", fc=BOXC[k], ec="none"))
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_color("#475569"); sp_.set_linewidth(0.8)
        stage_info(0.06, 1.72, sy, margin(p0), c["tau_g"], opt(a0), a0 == r["answer"], t0, s0,
                   "stop" if stage == 0 else "go")
        # (2) verification: each candidate with its weight
        skipped = stage == 0
        cw, gap = 0.44, 0.086
        if skipped:
            skipped_box(2.0, img_bot + 0.06, 2.02, 0.7, "Skipped: the glance is confident,\nso no candidate is verified")
        for k, b in enumerate(r["boxes"] if not skipped else []):
            x = 2.0 + k * (cw + gap)
            a = A(x, img_bot + 0.16, cw, cw)
            a.imshow(img.crop(b).resize((160, 160)), alpha=0.25 if skipped else 1.0)
            a.set_xticks([]); a.set_yticks([])
            sel = k in rd["chosen"] and not skipped
            for sp_ in a.spines.values():
                sp_.set_color(BOXC[k] if not skipped else "#CBD5E1"); sp_.set_linewidth(2.2 if sel else 0.8)
            a.text(0.03, 0.97, f"{k + 1}", transform=a.transAxes, color="white", fontsize=6.4, va="top",
                   fontweight="bold", bbox=dict(boxstyle="square,pad=0.08", fc=BOXC[k] if not skipped else "#CBD5E1",
                                                ec="none"))
            if skipped:
                continue
            w = rd["w"][k]
            by = img_bot + 0.06
            fig.add_artist(patches.Rectangle((x / FW, by / FH), cw / FW, 0.06 / FH, transform=fig.transFigure,
                                             fc=FADE, ec="none"))
            fig.add_artist(patches.Rectangle((x / FW, by / FH), cw * w / FW, 0.06 / FH, transform=fig.transFigure,
                                             fc=BOXC[k], ec="none"))
            T(x + cw / 2, img_bot + 0.62, f"w = {w:.2f}", fontsize=6.8, ha="center", va="bottom", fontweight="bold",
              color=INK if sel else MUTED)
        stage_info(2.0, 2.02, sy, margin(p1), c["tau_f"], opt(a1), a1 == r["answer"], t1, s1,
                   "off" if stage == 0 else ("stop" if stage == 1 else "go"))
        # (3) the zoomed regions, sizes proportional to their tokens
        if stage == 2:
            tmax = max(rd["tokens"])
            sides = [(t / tmax) ** 0.5 for t in rd["tokens"]]
            sc = min(0.7, (0.98 - 0.05 * (len(sides) - 1)) / sum(sides))
            sides = [sc * q for q in sides]
            xx = 4.24 + (0.98 - sum(sides) - 0.05 * (len(sides) - 1)) / 2
            for i, k in sorted(enumerate(rd["chosen"]), key=lambda z: -rd["tokens"][z[0]]):
                a = A(xx, img_bot + 0.06 + (0.7 - sides[i]) / 2, sides[i], sides[i])
                a.imshow(img.crop(r["boxes"][k]))
                a.set_xticks([]); a.set_yticks([])
                for sp_ in a.spines.values():
                    sp_.set_color(BOXC[k]); sp_.set_linewidth(2.0)
                xx += sides[i] + 0.05
        else:
            skipped_box(4.24, img_bot + 0.06, 0.98, 0.7,
                        "Skipped:\nthe verified\nanswer is\nconfident" if stage == 1 else "Skipped:\nthe glance\nis confident")
        stage_info(4.24, 0.98, sy, max(p2), None, opt(a2, 10), a2 == r["answer"], t2, s2,
                   "final" if stage == 2 else "off")
        # (4) answer and cost next to direct zooming and the full image (same question, same GPU)
        v = vic["vicrop-1"]
        ours_sec = s0 + (s1 if stage > 0 else 0) + (s2 if stage > 1 else 0)
        ours_tok = t0 + (t1 if stage > 0 else 0) + (t2 if stage > 1 else 0)
        meths = [(NAME, pred, ours_sec, ours_tok, OURS),
                 ("ViCrop", int(np.argmax(v["probs"])), v["seconds"], vic["glance"]["tokens"] + v["tokens"], "#64748B"),
                 ("Full image", int(np.argmax(nat["probs"])), nat["seconds"], nat["tokens"], "#64748B")]
        smax = max(m[2] for m in meths)
        X0 = 5.44
        for j, (name, ans, sec, tok, col) in enumerate(meths):
            yy = y0 + RH - 0.48 - j * 0.36
            ok = ans == r["answer"]
            T(X0, yy, name, fontsize=7.4, fontweight="bold", va="center", color=col if j == 0 else INK)
            T(X0 + 1.68, yy, f"{opt(ans, 11)} {'✓' if ok else '✗'}", fontsize=7.4, fontweight="bold", va="center",
              ha="right", color=GOOD if ok else BAD)
            bw = 0.62 * sec / smax
            fig.add_artist(patches.Rectangle((X0 / FW, (yy - 0.17) / FH), bw / FW, 0.07 / FH,
                                             transform=fig.transFigure, fc=col, ec="none"))
            T(X0 + bw + 0.05, yy - 0.135, f"{sec:.2f} s, {tok:,} tok", fontsize=6.8, fontweight="bold", va="center",
              color=MUTED)
    save(fig, "trace")


SEL_HEADS = {"3b": [(24, 5), (22, 10), (22, 0)], "7b": [(19, 22), (19, 15), (19, 17)]}
HEAD_EXAMPLES = ["direct_attributes/sa_4203", "direct_attributes/sa_50442"]


def insight():
    """(a) attention of the selected heads and of all heads on two V* questions, (b) attention around the annotated
    target over all of V*, (c, d) accuracy against the calibrated confidence of the glance and the verification"""
    from fovea_cascade_an import margin, temp
    from matplotlib.colors import LinearSegmentedColormap
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95), gridspec_kw={"wspace": 0.5, "width_ratios": [1.5, 1.05, 1, 1]})
    # (a) where the selected heads look: two V* questions, the attention of the question words of the selected heads
    # and of all heads (3B), cropped around the annotated target (white box). (b) the same over all of V*: attention
    # per glance cell against its distance to the target, relative to a uniform attention. scripts/head_target_mass.py
    def overlap_grid(boxes, W, H, R, C):          # as scripts/head_target_mass.py
        ys, xs = torch.linspace(0, H, R + 1), torch.linspace(0, W, C + 1)
        cover = torch.zeros(R, C)
        for b in boxes:
            oy = (torch.minimum(ys[1:], torch.tensor(float(b[1] + b[3]))) - torch.maximum(ys[:-1], torch.tensor(float(b[1])))).clamp_min(0)
            ox = (torch.minimum(xs[1:], torch.tensor(float(b[0] + b[2]))) - torch.maximum(xs[:-1], torch.tensor(float(b[0])))).clamp_min(0)
            cover += oy[:, None] * ox[None, :] / ((H / R) * (W / C))
        return cover.clamp(max=1)

    S = {s["id"]: s for s in LOADERS["vstar"][0](ROOT / LOADERS["vstar"][1])}
    HT = {m: torch.load(RUN / "vstar" / f"head_target_mass_{m}.pt") for m in ("3b", "7b")
          if (RUN / "vstar" / f"head_target_mass_{m}.pt").exists()}
    ax = axes[0]
    ax.axis("off")
    header(ax, "(a) Head attention, 3B")
    d = HT["3b"]
    pb0 = ax.get_position()
    gap = 0.012
    tw, th = (pb0.width - gap) / 2, (pb0.height - gap) / 2
    for i, qid in enumerate(HEAD_EXAMPLES):
        s = S[qid]
        img = Image.open(s["image"]).convert("RGB")
        W, H = img.size
        m = d["maps"][qid].float()
        bx = s["bboxes"][0]
        cx, cy = bx[0] + bx[2] / 2, bx[1] + bx[3] / 2
        asp = (tw * fig.get_figwidth()) / (th * fig.get_figheight())     # width / height of a thumbnail
        hw = min(max(4 * max(bx[2], bx[3]), 0.1 * W), W / 2, asp * H / 2)
        hh = hw / asp
        cx, cy = min(max(cx, hw), W - hw), min(max(cy, hh), H - hh)       # window inside the image
        pair = ((m[[l * d["H"] + h for l, h in SEL_HEADS["3b"]]].mean(0), "Selected heads"), (m.mean(0), "All heads"))
        vmax = None
        for j, (mp, lab) in enumerate(pair):
            a = fig.add_axes([pb0.x0 + j * (tw + gap), pb0.y1 - (i + 1) * th - i * gap, tw, th])
            small = np.asarray(img.resize((900, int(900 * H / W)))).astype(float) / 255
            sp = torch.nn.functional.interpolate(mp[None, None], size=small.shape[:2], mode="bilinear",
                                                 align_corners=False)[0, 0].numpy()
            if vmax is None:                  # one scale for both maps (each sums to 1): set by the selected heads
                vmax = np.percentile(sp, 99.5)
            sp = np.clip(sp / vmax, 0, 1) ** 0.6
            a.imshow(small * (0.5 + 0.5 * sp[..., None]), extent=(0, W, H, 0))
            rgba = plt.get_cmap("jet")(sp)
            rgba[..., 3] = 0.5 * sp
            a.imshow(rgba, extent=(0, W, H, 0), interpolation="bilinear")
            for b in s["bboxes"]:
                a.add_patch(patches.Rectangle((b[0], b[1]), b[2], b[3], fill=False, ec="white", lw=1.1, ls=(0, (2, 1))))
            a.set_xlim(cx - hw, cx + hw)
            a.set_ylim(cy + hh, cy - hh)
            a.set_aspect("auto")
            a.set_xticks([]); a.set_yticks([])
            for sp_ in a.spines.values():
                sp_.set_color(OURS if j == 0 else "#64748B"); sp_.set_linewidth(1.4)
            if i == 1:
                a.text(0.5, -0.05, lab, transform=a.transAxes, ha="center", va="top", fontsize=5.8,
                       fontweight="bold", color=OURS if j == 0 else "#475569")
    # (b) attention around the target, relative to uniform, all V* questions
    ax = axes[1]
    D = np.arange(0, 9)

    def profile(dd, which):
        acc = np.zeros(len(D)); cnt = np.zeros(len(D))
        for qid in dd["ids"]:
            s = S[qid]
            m = dd["maps"][qid].float()
            R, C = m.shape[1:]
            W, H = Image.open(s["image"]).size
            cover = overlap_grid(s["bboxes"], W, H, R, C) > 0
            rr, cc = torch.nonzero(cover, as_tuple=True)
            gr, gc = torch.meshgrid(torch.arange(R), torch.arange(C), indexing="ij")
            dist = torch.maximum((gr[..., None] - rr).abs(), (gc[..., None] - cc).abs()).min(-1).values
            if which == "sel":
                mp = m[[l * dd["H"] + h for l, h in SEL_HEADS[dd["tag"]]]].mean(0)
            elif which == "all":
                mp = m.mean(0)
            else:
                mp = m[(dd["L"] - 1) * dd["H"]:].mean(0)
            mp = mp * R * C                                   # 1 = uniform attention
            for k, dv in enumerate(D):
                sel = dist == dv
                if sel.any():
                    acc[k] += float(mp[sel].mean()); cnt[k] += 1
        return acc / cnt

    for m, ls, mk in (("3b", "-", "o"), ("7b", "--", "s")):
        if m not in HT:
            continue
        dd = dict(HT[m], tag=m)
        for which, col, lab in (("sel", OURS, "Selected heads"), ("all", "#64748B", "All heads")):
            ys = profile(dd, which)
            print("head profile", m, which, np.round(ys, 2))
            ax.plot(D, ys, ls, marker=mk, color=col, lw=1.6, ms=3.2, mec="black", mew=0.4,
                    mfc=col if m == "3b" else "white", zorder=4 if which == "sel" else 3)
    ax.axhline(1, color="#94A3B8", lw=0.8, ls=":", zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(0.3, 200)
    logticks(ax, "y", choices=(1, 10, 100))
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_xlabel("Distance to target", labelpad=1.5)
    ax.set_ylabel("Attention / uniform (log)", labelpad=1.5)
    tidy(ax)
    header(ax, "(b) Around the target")
    LEG_Y = -0.36                   # top of every legend row, in axes units
    axes[0].legend(handles=[matplotlib.lines.Line2D([], [], color=OURS, lw=1.6, label="Selected"),
                            matplotlib.lines.Line2D([], [], color="#64748B", lw=1.6, label="All heads"),
                            matplotlib.lines.Line2D([], [], color="#94A3B8", lw=0.9, ls=":", label="Uniform"),
                            matplotlib.lines.Line2D([], [], color="#333333", lw=1.2, marker="o", ms=3, label="3B"),
                            matplotlib.lines.Line2D([], [], color="#333333", lw=1.2, ls="--", marker="s", ms=3,
                                                    mfc="white", label="7B")],
                   loc="upper center", bbox_to_anchor=(0.95, LEG_Y), ncol=5, fontsize=5.8, handlelength=1.6,
                   columnspacing=0.8, handletextpad=0.3)
    # (c, d) accuracy of the glance and the verified answer against their calibrated confidence (lines) and the share
    # of questions in each confidence bin (bars), all four benchmarks
    bins = np.linspace(0, 1, 6)
    mid = (bins[:-1] + bins[1:]) / 2
    for ax, (model, tag, col, title) in zip(axes[2:], (("3b", "test_h", OURS, "(c) Confidence, 3B"),
                                                       ("7b", "test_7b", "#4338CA", "(d) Confidence, 7B"))):
        cf = config(model)
        rows = [r for ds in SETS.values() for r in rows_of(f"cascade_{tag}.jsonl", ds)]
        rd = [r["read"][cf["rule"]] for r in rows]
        st = {"Glance": (np.array([margin(temp(r["p0"], cf["Tg"])) for r in rows]),
                         np.array([int(np.argmax(r["p0"])) == r["answer"] for r in rows])),
              "Verify": (np.array([margin(temp(x["p1"], cf["Tf"])) for x in rd]),
                         np.array([int(np.argmax(x["p1"])) == r["answer"] for x, r in zip(rd, rows)]))}
        for j2, (name, (cs, ok)) in enumerate(st.items()):
            idx = [(cs >= lo) & ((cs < hi) | (hi == 1)) for lo, hi in zip(bins[:-1], bins[1:])]
            share = [100 * m.mean() for m in idx]
            ys = [100 * ok[m].mean() for m in idx]
            print(model, name, "share", [round(x, 1) for x in share], "acc", [round(y, 1) for y in ys])
            ax.bar(mid + (j2 - 0.5) * 0.085, share, width=0.08, color="#CBD5E1" if j2 == 0 else "#E0E7FF",
                   edgecolor="#64748B", lw=0.5, zorder=2, hatch="" if j2 == 0 else "////")
            ax.plot(mid, ys, "-" if j2 == 0 else "--", marker="o" if j2 == 0 else "s", color=col, lw=1.7, ms=3.8,
                    mec="black", mew=0.5, mfc=col if j2 == 0 else "white", zorder=4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xlabel("Calibrated confidence", labelpad=1.5)
        ax.set_ylabel("Accuracy / questions (%)", labelpad=1.5)
        tidy(ax)
        header(ax, title)
    axes[2].legend(handles=[matplotlib.lines.Line2D([], [], color="#333333", lw=1.5, marker="o", ms=3.5, mec="black",
                                                    mew=0.5, label="Glance acc."),
                            matplotlib.lines.Line2D([], [], color="#333333", lw=1.5, ls="--", marker="s", ms=3.5,
                                                    mec="black", mew=0.5, mfc="white", label="Verify acc."),
                            patches.Patch(fc="#CBD5E1", ec="#64748B", lw=0.5, label="Glance share"),
                            patches.Patch(fc="#E0E7FF", ec="#64748B", lw=0.5, hatch="////", label="Verify share")],
                   loc="upper center", bbox_to_anchor=(1.25, LEG_Y), ncol=4, fontsize=5.8, handlelength=1.6,
                   columnspacing=1.0, handletextpad=0.4)
    save(fig, "insight")


# ------------------------------------------------------------------------------------------------ newer models
CROSS = [("q3vl", "Qwen3-VL-8B"), ("q35", "Qwen3.5-9B")]
RUNP = ROOT / "runs_pro"          # results of the RTX PRO 6000 (server/sync_results.sh)


def rows_pro(name, ds):
    paths = [RUNP / d / name for d in ds]
    if not all(p.exists() for p in paths):
        return None
    rows = sum((load(p) for p in paths), [])
    full = {"V*": 191, "HR": 800, "MME": 1919}["V*" if ds[0] == "vstar" else "MME" if ds[0] == "mmerw" else "HR"]
    return rows if len(rows) == full else None


def cross_numbers():
    """{model: {method: {set: (acc, seconds)}}}, accuracy and latency all from the RTX PRO 6000"""
    out = {}
    for tag, title in CROSS:
        cfg_path = RUNP / "mmerwtrain" / f"cascade_{tag}_config.json"
        if not cfg_path.exists():
            continue
        c = json.load(open(cfg_path))
        R = {}
        for st, ds in SETS.items():
            rows = rows_pro(f"cascade_test_{tag}.jsonl", ds)
            if rows:
                a, s, _ = cascade(rows, c["rule"], c["refine"], c["Tg"], c["Tf"], c["tau_g"], c["tau_f"])
                R.setdefault(NAME, {})[st] = (a, s)
                R.setdefault("Glance (1,024 tokens)", {})[st] = (
                    100 * np.mean([int(np.argmax(r["p0"])) == r["answer"] for r in rows]),
                    float(np.mean([r["seconds"]["glance"] for r in rows])))
            rows = rows_pro(f"native_{tag}.jsonl", ds)
            if rows:
                for t, lab in (("1600", "Downscale (1,600 tokens)"), ("8192", "Full image (8,192 tokens)")):
                    if t in rows[0]:
                        R.setdefault(lab, {})[st] = (100 * np.mean([r[t]["correct"] for r in rows]),
                                                     float(np.mean([r[t]["seconds"] for r in rows])))
            rows = rows_pro(f"e2e_fast_{tag}.jsonl", ds)
            if rows and "vicrop-1" in rows[0]:
                R.setdefault("ViCrop", {})[st] = (100 * np.mean([r["vicrop-1"]["correct"] for r in rows]),
                                                  float(np.mean([r["vicrop-1"]["seconds"] for r in rows])))
        # latency: separate runs on the A800 with the GPU to themselves (/tmp/lat_run.sh), same questions for all
        # methods (every 2nd of V* / HR-Bench, every 19th of MME-RealWorld), replaces the seconds of the runs above
        lat = {}
        for st, ds in SETS.items():
            d = ds[0]
            files = {k: RUNP / d / f"{k}_lat_{tag}.jsonl" for k in ("cascade", "native", "e2e")}
            if not all(f.exists() for f in files.values()):
                continue
            rc, rn, re_ = (load(files[k]) for k in ("cascade", "native", "e2e"))
            if not (len(rc) == len(rn) == len(re_)) or [r["id"] for r in rc] != [r["id"] for r in rn]:
                continue
            lat.setdefault(NAME, {})[st] = cascade(rc, c["rule"], c["refine"], c["Tg"], c["Tf"], c["tau_g"],
                                                   c["tau_f"])[1]
            lat.setdefault("Glance (1,024 tokens)", {})[st] = float(np.mean([r["seconds"]["glance"] for r in rc]))
            lat.setdefault("Downscale (1,600 tokens)", {})[st] = float(np.mean([r["1600"]["seconds"] for r in rn]))
            lat.setdefault("Full image (8,192 tokens)", {})[st] = float(np.mean([r["8192"]["seconds"] for r in rn]))
            lat.setdefault("ViCrop", {})[st] = float(np.mean([r["vicrop-1"]["seconds"] for r in re_]))
            # the latency runs must reproduce the answers of the accuracy runs on the same questions
            full = {r["id"]: r for r in rows_pro(f"cascade_test_{tag}.jsonl", ds)}
            same = np.mean([int(np.argmax(r["p0"])) == int(np.argmax(full[r["id"]]["p0"])) for r in rc])
            print(f"{title} {st}: {len(rc)} latency questions, glance answer as in the accuracy run {100 * same:.1f}%")
        if lat and all(len(v) == len(SETS) for v in lat.values()):
            for m in R:
                R[m] = {st: (a, lat[m][st]) for st, (a, _) in R[m].items()}
            R["_lat"] = True
        out[title] = R
        for m, v in R.items():
            if m != "_lat":
                print(f"{title:16s} {m:26s} " + " ".join(f"{st} {a:5.1f}/{s:.2f}s" for st, (a, s) in v.items()))
    return out


def cross_table():
    N = cross_numbers()
    sets = list(SETS)
    order = ["Glance (1,024 tokens)", "Downscale (1,600 tokens)", "Full image (8,192 tokens)", "ViCrop", NAME]
    body = []
    for title, R in N.items():
        body.append(r"    \multicolumn{7}{l}{\textit{" + title + r"}}\\")
        methods = [m for m in order if m in R]
        cols = []
        for st in sets:
            cols.append(fmt_col([R[m][st][0] if st in R[m] else None for m in methods]))
        avg = [np.mean([R[m][st][0] for st in sets]) if all(st in R[m] for st in sets) else None for m in methods]
        cols.append(fmt_col(avg))
        # accuracy runs shared their GPU with other jobs: only the dedicated A800 latency runs are reported
        assert R.get("_lat"), f"{title}: latency runs missing"
        lat = [np.mean([R[m][st][1] for st in sets]) for m in methods]
        for i, m in enumerate(methods):
            label = r"\name{}" if m == NAME else ("ViCrop~\\cite{zhang2025vicrop}" if m == "ViCrop" else m)
            row = f"    {label} & " + " & ".join(col[i] for col in cols) + \
                  (f" & {lat[i]:.2f}" if lat[i] is not None else " & --") + r" \\"
            body.append((r"    \rowcolor{oursbg}" + row[4:]) if m == NAME else row)
        body.append(r"    \midrule")
    body = body[:-1]
    tex = (r"""\begin{table}[t]
  \centering
  \caption{Accuracy (\%) and mean latency (s) with newer models. Latency is measured on one A800 GPU on the same
  questions for all methods, every second question of V* and HR-Bench and every 19th of MME-RealWorld. The heads,
  exits and read form of \name{} are selected on the training split as for Qwen2.5-VL.}
  \label{tab:cross}
  \footnotesize
  \setlength{\tabcolsep}{3.2pt}
  \begin{tabular}{l ccccc c}
    \toprule
    Method & V* & 4K & 8K & MME & Avg. & Sec. \\
    \midrule
""" + "\n".join(body) + r"""
    \bottomrule
  \end{tabular}
\end{table}
""")
    (ROOT / "outputs" / "tab" / "cross.tex").write_text(tex, encoding="utf-8")
    print(tex)


# ------------------------------------------------------------------------------------------------ open-ended
OPEN_SETS = [("V*", "vstar"), ("4K", "hrbench4k"), ("8K", "hrbench8k"), ("MME", "mmerw")]


def open_table():
    """accuracy (%) of the open-ended subsets: runs/fovea/open_summary{,_7b}.json (scripts/fovea_open_select.py) and
    the HiDe / AttWarp open runs of the RTX PRO 6000 (runs_pro), on the same questions"""
    from fovea_open import answerable
    blocks = []
    for tag, title in (("", "Qwen2.5-VL-3B"), ("_7b", "Qwen2.5-VL-7B")):
        summ = json.load(open(RUN / f"open_summary{tag}.json"))["sets"]
        methods = [("native1600", "Downscale (1,600 tokens)"), ("native8192", "Full image (8,192 tokens)"),
                   ("vicrop", "ViCrop~\\cite{zhang2025vicrop}"), ("AttWarp", "AttWarp~\\cite{dalal2026attwarp}"),
                   ("HiDe", "HiDe~\\cite{liu2026hide}"), (NAME, r"\name{}")]
        acc = {}
        for key, _ in methods:
            for st, ds in OPEN_SETS:
                if key in ("HiDe", "AttWarp") and tag == "":
                    ids = [json.loads(l)["id"] for l in open(RUN / ds / "open.jsonl", encoding="utf-8")]
                    ids = [i for i in ids if i]
                    name = "hide_open.jsonl" if key == "HiDe" else "attwarp_open.jsonl"
                    rows = [json.loads(l) for l in open(RUN / ds / "open.jsonl", encoding="utf-8")]
                    rows = [r for r in rows if answerable(r)]
                    for base in (RUN, RUNP):     # local RTX 5080 run first, then the RTX PRO 6000 copy
                        f = base / ds / name
                        other = {x["id"]: x for x in load(f)} if f.exists() else {}
                        if other and all(r["id"] in other for r in rows):
                            acc[(key, st)] = 100 * np.mean([other[r["id"]]["correct"] for r in rows])
                            break
                elif key in summ.get(ds, {}):
                    acc[(key, st)] = summ[ds][key]["acc"]
        present = [(k, lab) for k, lab in methods if any((k, st) in acc for st, _ in OPEN_SETS)]
        cols = [fmt_col([acc.get((k, st)) for k, _ in present]) for st, _ in OPEN_SETS]
        avg = [np.mean([acc[(k, st)] for st, _ in OPEN_SETS]) if all((k, st) in acc for st, _ in OPEN_SETS) else None
               for k, _ in present]
        cols.append(fmt_col(avg))
        blocks.append(r"    \multicolumn{6}{l}{\textit{" + title + r"}}\\")
        for i, (k, lab) in enumerate(present):
            row = f"    {lab} & " + " & ".join(c[i] for c in cols) + r" \\"
            blocks.append((r"    \rowcolor{oursbg}" + row[4:]) if k == NAME else row)
        blocks.append(r"    \midrule")
        print(title, {f"{k}/{st}": round(v, 1) for (k, st), v in acc.items()})
    n = {st: len([1 for l in open(RUN / ds / "open.jsonl", encoding="utf-8") if answerable(json.loads(l))])
         for st, ds in OPEN_SETS}
    tex = (r"""\begin{table}[t]
  \centering
  \caption{Open-ended accuracy (\%): the options are removed and the generated answer is matched to the correct
  option. Questions per set: V* """ + str(n["V*"]) + ", HR-Bench 4K " + str(n["4K"]) + ", 8K " + str(n["8K"]) +
           ", MME-RealWorld " + str(n["MME"]) + r""".}
  \label{tab:open}
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{l ccccc}
    \toprule
    Method & V* & 4K & 8K & MME & Avg. \\
    \midrule
""" + "\n".join(blocks[:-1]) + r"""
    \bottomrule
  \end{tabular}
\end{table}
""")
    (ROOT / "outputs" / "tab" / "open.tex").write_text(tex, encoding="utf-8")
    print(tex)



def open_final(r, cfg):
    """answer text and correctness of the open-ended cascade (fovea_open_select.cascade, as reported)"""
    from fovea_open import norm
    o = r["ours"]
    e = o["read"]["both"]
    if o["c0"] >= cfg["tau_g"]:
        return o["a0"], o["ok0"], 0
    if e["c1"] >= cfg["tau_f"] and norm(e["a1"]) == norm(o["a0"]):
        return e["a1"], e["ok1"], 1
    return e[cfg["form"]]["a2"], e[cfg["form"]]["ok2"], 2


def qualitative_open():
    """Fig.: open-ended answers of every method on one question per benchmark (3B), glance wrong, VeriZoom right"""
    import textwrap
    from matplotlib.patches import ConnectionPatch
    from fovea_open import answerable, open_samples
    cfg = json.load(open(RUN / "mmerwtrain" / "open_config.json"))
    FW, FH, CW = 7.0, 1.95, 7.0 / 4
    fig = plt.figure(figsize=(FW, FH))
    A = lambda x, y, w, h: fig.add_axes([x / FW, y / FH, w / FW, h / FH])
    T = lambda x, y, t, **kw: fig.text(x / FW, y / FH, t, **kw)
    for col, (title, ds) in enumerate(QUAL_SETS_OPEN):
        S = {s["id"]: s for s in open_samples(ds)}
        cands = []
        for r in load(RUN / ds / "open.jsonl"):
            if not answerable(r) or r["id"] not in S:
                continue
            ans, ok, st = open_final(r, cfg)
            if not ok or r["ours"]["ok0"] or st == 0 or len(S[r["id"]]["question"]) > 90 or len(r["gold"]) > 18:
                continue
            W, H = Image.open(S[r["id"]]["image"]).size
            if not 1.2 <= W / H <= 2.2:
                continue
            wrong = (not r["native8192"]["correct"]) + (not r["vicrop"]["correct"])
            cands.append((-wrong, len(S[r["id"]]["question"]), r, ans, st))
        cands.sort(key=lambda z: (z[0], z[1]))
        _, _, r, ans, st = cands[QUAL_OPEN_PICK.get(ds, 0)]
        s = S[r["id"]]
        print(title, s["id"], "full", r["native8192"]["correct"], "vicrop", r["vicrop"]["correct"], len(cands))
        img = Image.open(s["image"]).convert("RGB")
        W, H = img.size
        x0, uw = col * CW + 0.05, CW - 0.1
        fig.add_artist(patches.FancyBboxPatch(((x0 - 0.03) / FW, 0.02 / FH), (uw + 0.06) / FW, (FH - 0.04) / FH,
                                              boxstyle="round,pad=0,rounding_size=0.012", fc="none", ec="#475569",
                                              lw=1.8, transform=fig.transFigure, zorder=-10))
        T(x0 + uw / 2, FH - 0.13, title, fontsize=7.2, color="white", ha="center", va="center", fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.3", fc="#334155", ec="none"))
        a = W / H
        h = min(0.78, (uw - 0.06) / (a + 0.62))
        side = 0.62 * h
        ytop = FH - 0.27
        ax = A(x0, ytop - h, h * a, h)
        ax.imshow(img.resize((900, max(1, int(900 / a)))), extent=(0, W, H, 0))
        rd = r["ours"]["read"]["both"]
        k = rd["chosen"][int(np.argmax([rd["w"][i] for i in rd["chosen"]]))]
        for j, b in enumerate(r["ours"]["boxes"]):
            sel = j in rd["chosen"]
            ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                                           ec=OURS if j == k else ("#D97706" if sel else "white"),
                                           lw=1.6 if sel else 0.8, ls="-" if sel else (0, (2, 1.5))))
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#475569"); sp.set_linewidth(1.2)
        b = r["ours"]["boxes"][k]
        crop = img.crop(b)
        m = max(crop.size)
        sq = Image.new("RGB", (m, m), (255, 255, 255))
        sq.paste(crop, ((m - crop.size[0]) // 2, (m - crop.size[1]) // 2))
        az = A(x0 + uw - side, ytop - h, side, side)
        az.imshow(sq)
        az.set_xticks([]); az.set_yticks([])
        for sp in az.spines.values():
            sp.set_color(OURS); sp.set_linewidth(2.2)
        for (px, py), (qx, qy) in (((b[2], b[1]), (0, 1)), ((b[2], b[3]), (0, 0))):
            fig.add_artist(ConnectionPatch((px, py), (qx, qy), coordsA="data", coordsB="axes fraction", axesA=ax,
                                           axesB=az, color=OURS, lw=1.6, ls=(0, (3, 1.8))))
        q = textwrap.fill(s["question"], 36)
        T(x0, ytop - h - 0.07, q, fontsize=5.6, color=INK, va="top", fontweight="bold", linespacing=1.15)
        T(x0, 0.53, f"gold: {r['gold']}", fontsize=5.4, color=MUTED, va="top", fontweight="bold")
        lines = [("Glance", r["ours"]["a0"], r["ours"]["ok0"], "#555555"),
                 ("Full image", r["native8192"]["answer"], r["native8192"]["correct"], "#333333"),
                 ("ViCrop", r["vicrop"]["answer"], r["vicrop"]["correct"], STYLE["ViCrop"][0]),
                 (NAME, ans, True, OURS)]
        ty = 0.42
        for lab, a_, ok, colr in lines:
            mark = "\u2713" if ok else "\u2717"
            T(x0, ty, f"{lab}:", fontsize=5.4, color=colr, va="top", fontweight="bold")
            T(x0 + 0.5, ty, textwrap.shorten(str(a_), 22, placeholder="...") + "  " + mark, fontsize=5.4,
              color=INK if ok else "#B91C1C", va="top", fontweight="bold")
            ty -= 0.1
    save(fig, "qualitative_open")


QUAL_SETS_OPEN = [("V* | open-ended", "vstar"), ("HR-Bench 4K | open-ended", "hrbench4k"),
                  ("HR-Bench 8K | open-ended", "hrbench8k"), ("MME-RealWorld | open-ended", "mmerw")]
QUAL_OPEN_PICK = {}


if __name__ == "__main__":
    for name in sys.argv[1:] or ["numbers"]:
        globals()[name]()

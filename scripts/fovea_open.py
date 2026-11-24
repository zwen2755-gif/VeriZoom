"""Open-ended versions of the benchmarks: the options are removed, the model answers freely and the answer is matched
to the text of the correct option after normalization. Only questions whose correct option is a short phrase (colours,
numbers, text, left/right, short names) are kept; options that deny the content and questions that refer to the
options are dropped.

    python scripts/fovea_open.py          # subset sizes per benchmark
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fovea_label import LOADERS  # noqa: E402

INSTRUCTION = "Answer the question using a single word or phrase."
NUM = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                       "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
DENY = re.compile(r"\b(not|none|no such|cannot|can't|unknown|unable|doesn'?t|does not|all the above|wrong)\b", re.I)
REFERS = re.compile(r"\b(options?|following|choices?|choose|select)\b", re.I)
OPEN_SETS = {"vstar": "vstar", "hrbench4k": "hrbench4k", "hrbench8k": "hrbench8k", "mmerw": "mmerw",
             "mmerwtrain": "mmerwtrain"}


def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)                # 1,000 -> 1000
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)                 # dots that are not decimal points
    s = re.sub(r"[^\w\s.%]", " ", s.replace("_", " "))
    words = [NUM.get(w, w) for w in s.split() if w not in ("a", "an", "the")]
    return " ".join(words)


def core(option, question=""):
    """the phrase an open answer should contain: V* options are sentences, left/right questions keep the side"""
    o = str(option).strip()
    low = o.lower()
    if re.search(r"left or (the )?right|right or (the )?left", question, re.I):
        if re.search(r"\bleft\b", low) and not re.search(r"\bright\b", low):
            return "left"
        if re.search(r"\bright\b", low) and not re.search(r"\bleft\b", low):
            return "right"
    if len(o) > 26 and " is " in o:
        o = o.rsplit(" is ", 1)[1]
    return o.rstrip(".")


def contains(hay, needle):
    return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", hay) is not None


def match(pred, gold, others):
    """exact match after normalization, or the gold phrase inside the answer while no other option is"""
    p, g = norm(pred), norm(gold)
    if not p:
        return False
    if p == g or p.replace(" ", "") == g.replace(" ", ""):
        return True
    if contains(p, g):
        return not any(contains(p, norm(o)) for o in others if norm(o) and norm(o) != g and not contains(g, norm(o)))
    return False


# MME-RealWorld driving behaviour questions answer with labels of a fixed taxonomy ("yielding", "no response",
# "accelerating/speeding up", ...) that a free-form answer cannot reproduce; they are left out of the open-ended sets
LABEL_TASKS = re.compile(r"Motion|Intention|intention|Interaction|Attention_TrafficSignal")


def answerable(sample):
    return not LABEL_TASKS.search(sample["id"] + " " + str(sample.get("category", "")))


def open_samples(dataset, exclude_labels=True):
    """samples of `dataset` with an open-ended prompt, the gold phrase and the other options' phrases (without the
    label-taxonomy questions unless exclude_labels=False, e.g. to subsample first and filter afterwards)"""
    if exclude_labels:
        return [s for s in open_samples(dataset, exclude_labels=False) if answerable(s)]
    loader, root = LOADERS[OPEN_SETS[dataset]]
    out = []
    for s in loader(ROOT / root):
        if REFERS.search(s["question"]):
            continue
        opts = [core(o, s["question"]) for o in s["options"]]
        gold = opts[s["answer"]]
        g = norm(gold)
        if (not g or len(g.split()) > 3 or len(g) > 24 or g in ("other", "others")
                or DENY.search(str(s["options"][s["answer"]]))):
            continue
        normed = [norm(o) for o in opts]
        if normed.count(g) > 1:
            continue
        out.append(dict(s, open_text=f"{s['question']}\n{INSTRUCTION}", gold=gold,
                        others=[o for k, o in enumerate(opts) if k != s["answer"]]))
    return out


if __name__ == "__main__":
    from collections import Counter
    for ds in OPEN_SETS:
        S = open_samples(ds)
        cats = Counter(x["id"].split("/")[0] if ds != "mmerw" and ds != "mmerwtrain" else "/".join(x["id"].split("/")[:2])
                       for x in S)
        print(ds, len(S), dict(cats.most_common(8)))
        for x in S[:3]:
            print("   ", x["question"][:70], "->", x["gold"], "| others", x["others"][:3])
    assert match("On the left side.", "left", ["right"]) and not match("left or right", "left", ["right"])
    assert match("ISTRE.PULA", "ISTRE.PULA", []) and match("Three", "3", ["2"]) and match("1,250", "1250", [])
    print("match tests ok")

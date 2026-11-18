"""Hassaniya ASR scoring: a reimplementation of the scoring steps documented in MauriAI's
Hassaniya Speech Test v1 (`score_hassaniya.py`, Bewa / MauriAI 2026, CC BY-NC 4.0), so our
numbers are directly comparable with their published table. Verified to reproduce their
published Step 3 / Light figures (see data/benchmarks/asr/results.md).

The house spelling table (house_spelling.json) is MauriAI's and is fetched at runtime,
not redistributed here.

Steps (applied identically to reference and hypothesis):
  0 Light      NFC, harakat/tatweel removed, alif forms->ا, ى->ي, ة->ه, گ->ك
  1 signs      + punctuation/non-Arabic removed, hamza carriers folded, 3+ repeated letters -> 1
  2 spelling   + house spelling table, گ/ق/ك merged, hesitations dropped
  3 custom     step 2 text; CER ignores spaces; WER lets one word match two joined words for free
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

_STRIP = dict.fromkeys([0x64B, 0x64C, 0x64D, 0x64E, 0x64F, 0x650, 0x651, 0x652, 0x653, 0x654, 0x655, 0x670, 0x640], None)


def light(t: str) -> str:
    if not t:
        return ""
    t = unicodedata.normalize("NFC", t).translate(_STRIP)
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"), ("ة", "ه")):
        t = t.replace(a, b)
    return " ".join(t.split()).replace("گ", "ك")


def signs(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    t = re.sub("[ً-ٰٟـۖ-ۭ]", "", t)
    t = light(t)
    t = re.sub("[آأإٱ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    t = t.replace("ؤ", "و").replace("ئ", "ي").replace("ء", "")
    t = re.sub("[^ء-يگ0-9 ]", " ", t)
    t = re.sub(r"(.)\1{2,}", r"\1", t)
    return " ".join(t.split())


class House:
    AR = re.compile("[ء-يپچڤگکی]+")
    DIAC = re.compile("[ً-ْـ]")
    LONG = re.compile("([ء-يگ])\\1{2,}")
    FILL = re.compile(r"^(?:ا+مم+|ا*هه+|اا+ه?|مم+|اه)$")
    HEAVY = str.maketrans({"ة": "ه", "گ": "ك", "ق": "ك", "ؤ": "و", "ئ": "ي"})
    LIGHT = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا"})

    def __init__(self, table_path: Path):
        h = json.loads(Path(table_path).read_text(encoding="utf-8"))
        self.table, self.group, self.hold = h["table"], h["group_choice"], set(h["never_rewritten"])

    def _l(self, w):
        w = w.translate(self.LIGHT)
        return w[:-1] + "ي" if w.endswith("ى") else w

    def word(self, w):
        w = self.LONG.sub(lambda m: m.group(1) * (3 if m.group(1) == "ه" else 1), w)
        if w == "ألى":
            return "اللي"
        if w[0] in "أإ" and len(w) > 1:
            w = "ا" + w[1:]
        elif len(w) > 2 and w[0] in ("و", "ف") and w[1] in "أإ":
            w = w[0] + "ا" + w[2:]
        if w in self.table:
            return self.table[w]
        k = w.translate(self.HEAVY).replace("ء", "")
        if k in self.group and w not in self.hold and self._l(w) != self._l(self.group[k]) and w != self.group[k]:
            return self.group[k]
        return w

    def spelling(self, t: str) -> str:
        def rep(m):
            r = self.word(m.group(0))
            return r[:-1] + "ه" if r.endswith("ة") else r
        t = signs(self.AR.sub(rep, self.DIAC.sub("", t)))
        t = t.replace("گ", "ك").replace("ق", "ك")
        return " ".join(w for w in t.split() if not self.FILL.match(w))


def edit(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer_join(r, h) -> int:
    n, m, inf = len(r), len(h), 10 ** 9
    D = [[inf] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            b = inf
            if i:
                b = min(b, D[i - 1][j] + 1)
            if j:
                b = min(b, D[i][j - 1] + 1)
            if i and j:
                b = min(b, D[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
            if i >= 2 and j and r[i - 2] + r[i - 1] == h[j - 1]:
                b = min(b, D[i - 2][j - 1])
            if j >= 2 and i and h[j - 2] + h[j - 1] == r[i - 1]:
                b = min(b, D[i - 1][j - 2])
            D[i][j] = b
    return D[n][m]


def score(pairs, table_path: Path) -> dict:
    """pairs: [(ref, hyp)] -> {'light': (cer, wer), 'step3': (cer, wer)} pooled over clips, in %."""
    house = House(table_path)
    out = {}
    for name, f, join in (("light", light, False), ("step3", house.spelling, True)):
        ce = nc = we = nw = 0
        for ref, hyp in pairs:
            a, b = f(ref), f(hyp or "")
            if join:
                ce += edit(a.replace(" ", ""), b.replace(" ", ""))
                nc += len(a.replace(" ", ""))
                we += wer_join(a.split(), b.split())
            else:
                ce += edit(a, b)
                nc += len(a)
                we += edit(a.split(), b.split())
            nw += len(a.split())
        out[name] = (100 * ce / max(nc, 1), 100 * we / max(nw, 1))
    return out

"""Shared Hassaniya language helpers: lexicon loading, normalization,
Arabizi transliteration, numbers/time formatting, gender agreement."""
from __future__ import annotations

import random
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LEX_PATH = ROOT / "data" / "lexicon" / "lexicon.yaml"


def load_lexicon(path: Path = LEX_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------- normalize
TASHKEEL = re.compile(r"[ً-ْٰـ]")  # harakat, shadda, sukun, tatweel


def strip_tashkeel(s: str) -> str:
    return TASHKEEL.sub("", s)


def normalize(s: str) -> str:
    """Canonical form used for matching/eval. Conservative: keeps گ distinct
    from ق because it is phonemic-ish in Hassaniya, but folds the variants
    ASR systems and keyboards actually produce."""
    s = strip_tashkeel(s)
    s = re.sub("[أإآٱ]", "ا", s)
    s = s.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    s = s.replace("ڭ", "گ").replace("ݣ", "گ").replace("ڨ", "گ").replace("كْ", "ك")
    s = s.replace("پ", "ب").replace("ڤ", "ف")
    s = re.sub(r"[؟?!.,،؛:\"'«»]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Noise that a real Arabic ASR (Whisper etc.) or a phone keyboard produces on Hassaniya
_ASR_SUBS = [
    ("گ", "ق"), ("گ", "ك"), ("گ", "ج"), ("ة", "ه"), ("ي", "ى"), ("ذ", "د"),
    ("ظ", "ض"), ("ض", "ظ"), ("ث", "ت"), ("ڭ", "گ"),
]


def asr_noise(s: str, rng: random.Random, p: float = 0.35) -> str:
    s = strip_tashkeel(s) if rng.random() < 0.8 else s
    for a, b in _ASR_SUBS:
        if a in s and rng.random() < p * 0.4:
            s = s.replace(a, b)
    # glue/unglue proclitic prepositions: "ف الصالون" <-> "فالصالون"
    if rng.random() < p:
        s = re.sub(r"\bف ال", "فال", s)
    if rng.random() < p * 0.5:
        s = s.replace("ال", "ل", 1) if s.startswith("ال") else s
    # drop punctuation, collapse
    s = re.sub(r"[؟?!.,،]", "", s)
    return re.sub(r"\s+", " ", s).strip()


# ----------------------------------------------------------------- Arabizi
_AZ = {
    "ا": "a", "أ": "a", "إ": "e", "آ": "a", "ٱ": "a", "ب": "b", "ت": "t", "ث": "th",
    "ج": "j", "ح": "7", "خ": "kh", "د": "d", "ذ": "dh", "ر": "r", "ز": "z", "س": "s",
    "ش": "ch", "ص": "s", "ض": "d", "ط": "t", "ظ": "dh", "ع": "3", "غ": "gh", "ف": "f",
    "ق": "9", "گ": "g", "ڭ": "g", "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "h",
    "و": "w", "ي": "y", "ة": "a", "ى": "a", "ء": "2", "ؤ": "ou", "ئ": "2i",
    "پ": "p", "ڤ": "v", "؟": "?", "،": ",",
}


def to_arabizi(s: str) -> str:
    """Rough, rule-based Arabic-script -> Arabizi (chat alphabet) conversion.
    Good enough as an augmentation so the model also understands typed Latin
    Hassaniya; not a linguistic transcription."""
    s = strip_tashkeel(s.replace("ّ", "ّ"))
    out = []
    words = s.split()
    for w in words:
        pre = ""
        if w.startswith("ال") and len(w) > 2:
            pre, w = "el-", w[2:]
        elif w.startswith("وال") and len(w) > 3:
            pre, w = "wel-", w[3:]
        elif w.startswith("فال") and len(w) > 3:
            pre, w = "fel-", w[3:]
        buf = []
        for i, ch in enumerate(w):
            if ch == "و" and 0 < i < len(w) - 1 and w[i - 1] not in "اوي":
                buf.append("ou")
            elif ch == "و" and i == len(w) - 1 and i > 0:
                buf.append("ou")
            elif ch == "ي" and i > 0 and w[i - 1] not in "اوي":
                buf.append("i")
            else:
                buf.append(_AZ.get(ch, ch))
        out.append(pre + "".join(buf))
    return " ".join(out)


# ----------------------------------------------------------------- numbers
COUNT_SHORT = {3: "اثلاث", 4: "اربع", 5: "خمس", 6: "ستّ", 7: "سبع", 8: "اثمن", 9: "تسع", 10: "عشر"}


def num_word(n: int, lex: dict, rng: random.Random) -> str:
    nums = lex["numbers"]
    if n in nums:
        return rng.choice(nums[n])
    tens, unit = divmod(n, 10)
    if 1 <= unit and tens * 10 in nums and unit in nums:
        return f"{rng.choice(nums[unit])} و{rng.choice(nums[tens * 10])}"
    return str(n)


def say_number(n: int, lex: dict, rng: random.Random, allow_fr: bool = True) -> str:
    """How a speaker might say a number: Hassaniya word, digits, or French."""
    r = rng.random()
    fr = lex.get("numbers_fr", {})
    if allow_fr and n in fr and r < 0.15:
        return fr[n]
    if r < 0.35:
        return str(n)
    return num_word(n, lex, rng)


def count_with_unit(n: int, unit: str, rng: random.Random) -> str:
    """'5 minutes' with Arabic counted-noun agreement."""
    forms = {
        "minute": ("دقيقة", "دقيقتين", "دگايگ"),
        "hour": ("ساعة", "ساعتين", "ساعات"),
        "second": ("ثانية", "ثانيتين", "ثواني"),
    }[unit]
    sing, dual, plur = forms
    if n == 1:
        return sing
    if n == 2:
        return dual
    if 3 <= n <= 10:
        return f"{COUNT_SHORT[n] if rng.random() < 0.7 else n} {plur}"
    return f"{n} {sing}"


HOURS = {1: "واحدة", 2: "ثنتين", 3: "اثلاثة", 4: "اربعة", 5: "خمسة", 6: "ستّة", 7: "سبعة",
         8: "اثمانية", 9: "تسعة", 10: "عشرة", 11: "احداعش", 12: "اطناعش"}


def say_time(h: int, m: int) -> str:
    """Hassaniya/Maghrebi clock expression, e.g. 'اربعة ونصّ ف لعشيّة'."""
    period = ("ف الصباح" if 5 <= h < 12 else "ف لعشيّة" if 12 <= h < 18
              else "ف الليل")
    hh = h % 12 or 12
    if m == 0:
        t = f"{HOURS[hh]} ابّالظبط"
    elif m == 15:
        t = f"{HOURS[hh]} وربع"
    elif m == 30:
        t = f"{HOURS[hh]} ونصّ"
    elif m == 20:
        t = f"{HOURS[hh]} وثلث"
    elif m == 45:
        t = f"{HOURS[hh % 12 + 1]} الّا ربع"
    elif m > 45:
        t = f"{HOURS[hh % 12 + 1]} الّا {60 - m} دقيقة"
    else:
        t = f"{HOURS[hh]} و{m} دقيقة"
    return f"{t} {period}"


WEEKDAYS = ["الاثنين", "الثلاثاء", "لربعا", "الخميس", "الجمعة", "السبت", "الاحد"]
MONTHS = ["يناير", "فبراير", "مارس", "ابريل", "مايو", "يونيو", "يوليو", "اغشت",
          "سبتمبر", "اكتوبر", "نوفمبر", "دجمبر"]

# ----------------------------------------------------------------- agreement
PARTICIPLE = {  # state word, (masc, fem)
    "on": ("گادي", "گادية"),   # native speaker: گادي preferred over شاعل
    "off": ("مطفي", "مطفية"),   # native speaker: مطفي preferred, طافي also understood
    "open": ("مفتوح", "مفتوحة"),   # corpus: مفتوح 10, محلول 0
    "closed": ("مگفول", "مگفولة"),  # corpus: مگفول 14, مسدود 0
    "locked": ("مگفول", "مگفولة"),
    "unlocked": ("مفتوح", "مفتوحة"),
}


def pp(state: str, g: str) -> str:
    m, f = PARTICIPLE[state]
    return f if g == "f" else m


def strip_al(w: str) -> str:
    return w[2:] if w.startswith("ال") else w


def obj_suffix(g: str) -> str:
    return "ها" if g == "f" else "ه"

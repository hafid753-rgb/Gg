#!/usr/bin/env python3
"""Measure how well each lexicon form is attested in real Hassaniya text.

Uses the corpus built by scripts/ingest_external.py (falls back to the committed
permissive corpus). Matching is spelling-tolerant: MauriAI-style folding (harakat, hamza
carriers, ة/ه, ى/ي, گ/ق/ك merged). For forms with 0 hits it proposes attested spellings
with the same consonant skeleton (long vowels ا/و/ي dropped).

    python3 scripts/lexicon_evidence.py            # -> data/lexicon/evidence.json + docs/LEXICON_EVIDENCE.md
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hassaniya import ROOT, load_lexicon  # noqa: E402
from hsn_score import signs  # noqa: E402


def fold(t: str) -> str:
    return signs(t).replace("ق", "ك")


def skel(w: str) -> str:
    return re.sub("[اوي]", "", w) or w


def iter_forms(lex):
    """yield (section, key, form) for every Arabic-script surface form in the lexicon."""
    for vk, v in lex["verbs"].items():
        for f in ("imp_m", "imp_f", "imp_pl", "req"):
            for w in v.get(f, []):
                yield "verb", f"{vk}.{f}", w
    for sec in ("want", "now", "all", "very", "a_little", "this_m", "this_f", "that_m", "in", "of", "yes", "no", "ok"):
        for w in lex[sec]["ar"] if isinstance(lex[sec], dict) and "ar" in lex[sec] else []:
            yield "function", sec, w
    for q, v in lex["q"].items():
        for w in v["ar"]:
            yield "question", q, w
    for p in ("pre", "post"):
        for w in lex["politeness"][p]:
            yield "politeness", p, w
    for dk, d in lex["devices"].items():
        for it in d["items"]:
            for w in it["ar"]:
                yield "device", f"{dk}:{it['ar'][0]}", w
    for a in lex["areas"]:
        for w in a["ar"]:
            yield "area", a["id"], w
    for c in lex["colors"]:
        for w in c["ar"]:
            yield "color", c["id"], w
    for n, ws in lex["numbers"].items():
        for w in ws:
            yield "number", str(n), w
    for u, ws in lex["units"].items():
        for w in ws:
            yield "unit", u, w


def main():
    lex = load_lexicon()
    src = ROOT / "data/external/hassaniya_text_all.jsonl"
    if not src.exists():
        src = ROOT / "data/corpus/hassaniya_text_permissive.jsonl"
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    texts = [" " + fold(r["text"]) + " " for r in rows]
    tok = Counter(w for t in texts for w in t.split())
    by_skel = defaultdict(Counter)
    for w, c in tok.items():
        by_skel[skel(w)][w] += c
    # allow attached proclitics: و ف ب ل ال
    def hits(form: str) -> int:
        f = fold(form)
        if not f:
            return 0
        if " " in f:
            return sum(t.count(" " + f + " ") for t in texts)
        n = tok.get(f, 0)
        for p in ("و", "ف", "ب", "ل", "وال", "فال", "بال", "لل"):
            n += tok.get(p + f, 0) if len(f) > 2 else 0
        if f.startswith("ال"):
            n += tok.get("ل" + f[2:], 0)  # Hassaniya often writes the article as ل-
        return n

    report = []
    for sec, key, form in iter_forms(lex):
        h = hits(form)
        sugg = []
        if h == 0 and " " not in form:
            f = fold(form)
            sugg = [w for w, c in by_skel[skel(f)].most_common(3) if w != f]
            if f.startswith("ال"):
                sugg += [w for w, c in by_skel[skel("ل" + f[2:])].most_common(2) if w != f]
        report.append({"section": sec, "key": key, "form": form, "hits": h, "suggest": sugg[:4]})
    total = len(report)
    att = sum(r["hits"] > 0 for r in report)
    out = {"corpus": str(src.relative_to(ROOT)), "sentences": len(rows), "tokens": sum(tok.values()),
           "forms": total, "attested": att, "items": report}
    (ROOT / "data/lexicon/evidence.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    md = [f"# Lexicon evidence\n\nCorpus: {len(rows):,} real Hassaniya sentences ({sum(tok.values()):,} tokens) from the "
          f"sources in `data/corpus/SOURCES.md`. **{att}/{total} lexicon forms ({att / total:.0%}) are attested.** "
          "A zero count is not proof that a form is wrong: the corpus has little smart-home language. It marks the form "
          "for native-speaker review.\n",
          "| Section | Entry | Form | Hits | Attested look-alikes (same consonants) |", "|---|---|---|---|---|"]
    for r in sorted(report, key=lambda r: (r["section"], r["key"], -r["hits"])):
        md.append(f"| {r['section']} | {r['key']} | {r['form']} | {r['hits']} | {'، '.join(r['suggest'])} |")
    (ROOT / "docs/LEXICON_EVIDENCE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"attested {att}/{total} forms over {len(rows)} sentences")


if __name__ == "__main__":
    main()

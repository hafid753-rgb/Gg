#!/usr/bin/env python3
"""Collect existing public Hassaniya *text* into one corpus (standard library only).

Every row keeps its source and license. Two outputs:
  data/external/hassaniya_text_all.jsonl          ALL sources, incl. non-commercial / unknown-license
                                                  (git-ignored, local use only: lexicon evidence, eval)
  data/corpus/hassaniya_text_permissive.jsonl     only CC-BY-4.0 / Apache-2.0 / MIT sources, committed
                                                  with attribution (see data/corpus/SOURCES.md)

    python3 scripts/ingest_external.py
Needs network access to huggingface.co, datasets-server.huggingface.co and github.com.
"""
from __future__ import annotations

import csv
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALL = ROOT / "data" / "external" / "hassaniya_text_all.jsonl"
PERM = ROOT / "data" / "corpus" / "hassaniya_text_permissive.jsonl"
PERMISSIVE = {"CC-BY-4.0", "Apache-2.0", "MIT"}
UA = {"User-Agent": "hassaniya-dataset-builder/0.1"}


def get(url: str, tries: int = 4) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            print(f"  retry {url[:80]}: {e}", file=sys.stderr)
            time.sleep(2 * (i + 1))
    return b""


def hf_rows(repo: str, config="default", split="train"):
    off = 0
    while True:
        q = urllib.parse.urlencode({"dataset": repo, "config": config, "split": split, "offset": off, "length": 100})
        d = json.loads(get(f"https://datasets-server.huggingface.co/rows?{q}"))
        rows = d.get("rows", [])
        for r in rows:
            yield r["row"]
        off += len(rows)
        if not rows or off >= d.get("num_rows_total", 0):
            break


def hf_splits(repo: str):
    d = json.loads(get(f"https://datasets-server.huggingface.co/splits?dataset={urllib.parse.quote(repo)}"))
    return [(s["config"], s["split"]) for s in d.get("splits", [])]


def gh_raw(repo: str, path: str, branch: str = "main") -> bytes:
    return get(f"https://raw.githubusercontent.com/{repo}/{branch}/{urllib.parse.quote(path)}")


# (name, license, loader) ; loader yields dict(text=..., kind=..., extra...)
def src_dah():
    for r in hf_rows("hassan-IA/dah"):
        if (r.get("hassaniya-ar") or "").strip():
            yield {"text": r["hassaniya-ar"].strip(), "kind": "parallel", "en": r.get("english"),
                   "arabizi": r.get("hassaniya-en")}


def src_amthal():
    for r in hf_rows("ahmed02mk/amthal-hassaniya"):
        yield {"text": r["input"].strip(), "kind": "proverb", "gloss_msa": r.get("output")}


def src_mbarek():
    for r in hf_rows("mokhtar-46/mbarek_hassania"):
        yield {"text": r["input"].strip(), "kind": "dialog_user", "category": r.get("categorie")}
        yield {"text": r["output"].strip(), "kind": "dialog_reply_templated", "category": r.get("categorie")}


def src_text_cols(repo, cols, kind):
    def gen():
        for cfg, split in hf_splits(repo):
            for r in hf_rows(repo, cfg, split):
                for c in cols:
                    if isinstance(r.get(c), str) and r[c].strip():
                        yield {"text": r[c].strip(), "kind": kind, "split": split}
    return gen


def src_mauriai_test():
    for line in get("https://huggingface.co/datasets/MauriAI-lab/hassaniya-speech-test-v1/resolve/main/test.jsonl").decode().splitlines():
        r = json.loads(line)
        yield {"text": r["text"], "kind": "asr_transcript_broadcast", "id": r["id"]}


def src_mauriai_read():
    raw = get("https://huggingface.co/datasets/MauriAI-lab/hassaniya-read-speech-v1/resolve/main/audio/metadata.jsonl")
    for line in raw.decode().splitlines():
        r = json.loads(line)
        yield {"text": r["text"], "kind": "asr_prompt_read", "id": r["id"]}


def src_mauricorpus():
    for r in json.loads(gh_raw("rahma-boughale/MauriCorpus-data", "data/arabic_hassaniya.json")):
        yield {"text": r["hassaniya"], "kind": "parallel", "msa": r.get("arabic"), "en": r.get("english"), "fr": r.get("french")}


def src_opendata():
    repo = "mwimlmauritania-byte/Open-Data-Mauritania"
    for r in json.loads(gh_raw(repo, "datasets/hassaniya/hassaniya_stories_collection.json")):
        yield {"text": r["sentence"], "kind": "story_sentence"}
    for r in json.loads(gh_raw(repo, "datasets/multilingual/arabic_hassaniya_pulaar_trilingual.json")):
        yield {"text": r["hassanya"], "kind": "parallel", "msa": r.get("arabic")}


SOURCES = [
    # name, url, license, loader
    ("hassan-IA/dah", "https://huggingface.co/datasets/hassan-IA/dah", "CC-BY-4.0", src_dah),
    ("ahmed02mk/amthal-hassaniya", "https://huggingface.co/datasets/ahmed02mk/amthal-hassaniya", "CC-BY-4.0", src_amthal),
    ("mokhtar-46/mbarek_hassania", "https://huggingface.co/datasets/mokhtar-46/mbarek_hassania", "Apache-2.0", src_mbarek),
    ("MauriAI-lab/hassaniya-speech-test-v1", "https://huggingface.co/datasets/MauriAI-lab/hassaniya-speech-test-v1", "CC-BY-NC-4.0", src_mauriai_test),
    ("MauriAI-lab/hassaniya-read-speech-v1", "https://huggingface.co/datasets/MauriAI-lab/hassaniya-read-speech-v1", "CC-BY-NC-4.0", src_mauriai_read),
    ("rahma-boughale/MauriCorpus-data", "https://github.com/rahma-boughale/MauriCorpus-data", "CC-BY-NC-SA-4.0", src_mauricorpus),
    ("mwimlmauritania-byte/Open-Data-Mauritania", "https://github.com/mwimlmauritania-byte/Open-Data-Mauritania", "unclear (README CC-BY-4.0, Facebook-sourced text)", src_opendata),
    ("hadramibegnoug/abc.hassaniya_ASR", "https://huggingface.co/datasets/hadramibegnoug/abc.hassaniya_ASR", "unspecified",
     src_text_cols("hadramibegnoug/abc.hassaniya_ASR", ["transcription"], "asr_transcript")),
    ("Mamadou-Aw/Hassaniya-speech-dataset", "https://huggingface.co/datasets/Mamadou-Aw/Hassaniya-speech-dataset", "unspecified",
     src_text_cols("Mamadou-Aw/Hassaniya-speech-dataset", ["transcription"], "asr_transcript")),
    ("Mamadou-Aw/hassaniya-parler-tts", "https://huggingface.co/datasets/Mamadou-Aw/hassaniya-parler-tts", "unspecified",
     src_text_cols("Mamadou-Aw/hassaniya-parler-tts", ["text"], "tts_transcript")),
    ("Emin009/hassaniya-punctuation-dataset", "https://huggingface.co/datasets/Emin009/hassaniya-punctuation-dataset", "unspecified",
     src_text_cols("Emin009/hassaniya-punctuation-dataset", ["target"], "punctuated_text")),
]


def main():
    ALL.parent.mkdir(parents=True, exist_ok=True)
    PERM.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    counts = {}
    with open(ALL, "w", encoding="utf-8") as fa, open(PERM, "w", encoding="utf-8") as fp:
        for name, url, lic, loader in SOURCES:
            n = 0
            try:
                for row in loader():
                    key = (name, row["text"])
                    if key in seen or not row["text"]:
                        continue
                    seen.add(key)
                    rec = {"source": name, "license": lic, **row}
                    line = json.dumps(rec, ensure_ascii=False) + "\n"
                    fa.write(line)
                    if lic in PERMISSIVE:
                        fp.write(line)
                    n += 1
            except Exception as e:  # noqa: BLE001
                print(f"!! {name}: {e}", file=sys.stderr)
            counts[name] = (n, lic, url)
            print(f"{name:45s} {n:6d}  {lic}")
    (ROOT / "data" / "corpus" / "SOURCES.md").write_text(
        "# Hassaniya text corpus sources\n\n`hassaniya_text_permissive.jsonl` redistributes only the permissive rows below, "
        "unchanged apart from whitespace trimming and de-duplication. All other sources are fetched locally by "
        "`scripts/ingest_external.py` into the git-ignored `data/external/` and are used only to measure vocabulary "
        "(lexicon evidence) and for evaluation.\n\n| Source | Rows | License | Redistributed here |\n|---|---|---|---|\n"
        + "\n".join(f"| [{k}]({u}) | {n} | {lic} | {'yes' if lic in PERMISSIVE else 'no'} |" for k, (n, lic, u) in counts.items())
        + "\n\nCredit: please cite the original authors of each dataset when using this corpus.\n", encoding="utf-8")


if __name__ == "__main__":
    main()

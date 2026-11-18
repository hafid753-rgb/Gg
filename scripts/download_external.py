#!/usr/bin/env python3
"""Download the public Hassaniya resources we found (needs network access to huggingface.co).

  hassan-IA/dah                          Hassaniya<->English parallel text (Arabic script + Arabizi), from Tatoeba
  MauriAI-lab/hassaniya-read-speech-v1   ~4.4 h read speech, 900 clips, 5 speakers, CC-BY-NC-4.0
  MauriAI-lab/hassaniya-speech-test-v1   ~1 h broadcast speech, 408 clips, test only

Text from `dah` is exported to data/external/dah.jsonl; it is useful (a) to mine real
Hassaniya vocabulary into data/lexicon/lexicon.yaml, (b) as a small general-Hassaniya
mix-in for SFT so the model's replies stay dialectal. Always check each dataset's license
before redistribution: NC licenses forbid commercial use.

    pip install datasets soundfile && python3 scripts/download_external.py
"""
import json
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parents[1] / "data" / "external"
OUT.mkdir(parents=True, exist_ok=True)

TEXT = ["hassan-IA/dah"]
SPEECH = ["MauriAI-lab/hassaniya-read-speech-v1", "MauriAI-lab/hassaniya-speech-test-v1"]

for repo in TEXT:
    ds = load_dataset(repo)
    for split, d in ds.items():
        p = OUT / f"{repo.split('/')[-1]}.{split}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in d:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(repo, split, len(d), "->", p, "columns:", d.column_names)

for repo in SPEECH:
    ds = load_dataset(repo)  # cached under ~/.cache/huggingface; used directly by training/finetune_whisper.py
    print(repo, {k: len(v) for k, v in ds.items()}, "columns:", next(iter(ds.values())).column_names)

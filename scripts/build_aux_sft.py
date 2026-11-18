#!/usr/bin/env python3
"""Auxiliary SFT data from REAL Hassaniya text (permissive sources only), to keep the
fine-tuned model's Hassaniya authentic while it learns the synthetic smart-home tasks.

Tasks (from data/corpus/hassaniya_text_permissive.jsonl):
  * English -> Hassaniya translation              (hassan-IA/dah, CC-BY-4.0)
  * Hassaniya -> English meaning                  (hassan-IA/dah)
  * Arabizi (typed Latin Hassaniya) -> Arabic script (hassan-IA/dah)
  * explain a Hassaniya proverb (answer in Arabic) (ahmed02mk/amthal-hassaniya, CC-BY-4.0)

    python3 scripts/build_aux_sft.py   # -> data/generated/sft_aux/{train,dev}.jsonl
Mix at ~10-20% of training steps (training/finetune_llm_lora.py --aux-ratio 0.15).
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "corpus" / "hassaniya_text_permissive.jsonl"
OUT = ROOT / "data" / "generated" / "sft_aux"
SYS = ("You are \"Dar\" (دار), a Mauritanian assistant. You understand Hassaniya Arabic written in Arabic script or "
       "in Latin letters (Arabizi), and you write natural Hassaniya.")

EN2H = ["ترجم للحسانية: {x}", "كيف نگولو ذ بالحسانية: {x}", "Translate to Hassaniya: {x}", "{x}\nگولها بالحسانية"]
H2EN = ["شنه معنى ذ الكلام بالانگليزية: {x}", "What does this Hassaniya mean in English? {x}", "{x}\nترجمها للانگليزية"]
AZ2AR = ["اكتب ذ بالحرف العربي: {x}", "Write this Hassaniya in Arabic script: {x}", "{x}\nاكتبها بالعربي"]
PROV = ["شنه معنى المثل الحساني: {x}", "اشرح لي ذ المثل: {x}", "{x}\nشنه معنى ذ المثل؟"]


def msgs(user, answer):
    return [{"role": "system", "content": SYS}, {"role": "user", "content": user}, {"role": "assistant", "content": answer}]


def main():
    rng = random.Random(3)
    out = {"train": [], "dev": []}
    for line in open(SRC, encoding="utf-8"):
        r = json.loads(line)
        split = "dev" if int(hashlib.md5(r["text"].encode()).hexdigest(), 16) % 100 < 5 else "train"
        recs = []
        if r["source"] == "hassan-IA/dah":
            en, ha, az = (r.get("en") or "").strip(), r["text"].strip(), (r.get("arabizi") or "").strip()
            if en:
                recs.append(("en2hsn", msgs(rng.choice(EN2H).format(x=en), ha)))
                if rng.random() < 0.5:
                    recs.append(("hsn2en", msgs(rng.choice(H2EN).format(x=ha), en)))
            if az and rng.random() < 0.5:
                recs.append(("arabizi2ar", msgs(rng.choice(AZ2AR).format(x=az), ha)))
        elif r["source"] == "ahmed02mk/amthal-hassaniya" and r.get("gloss_msa"):
            recs.append(("proverb", msgs(rng.choice(PROV).format(x=r["text"]), r["gloss_msa"].strip())))
        for task, m in recs:
            out[split].append({"id": f"aux-{task}-{len(out[split]):05d}", "task": task, "source": r["source"],
                               "license": r["license"], "messages": m})
    OUT.mkdir(parents=True, exist_ok=True)
    for s, rows in out.items():
        with open(OUT / f"{s}.jsonl", "w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print({s: len(v) for s, v in out.items()})


if __name__ == "__main__":
    main()

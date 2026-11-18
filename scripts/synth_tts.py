#!/usr/bin/env python3
"""Synthesize Hassaniya smart-home command audio with an existing community TTS model.

STATUS (2026-11): the default voice is NOT intelligible enough. It gets 0% end-to-end through the three
best ASR models (data/benchmarks/asr/results_commands.md). Do not train on its output. Keep this
script for evaluating future, better Hassaniya TTS voices.

Default voice: mlemineb/mms-tts-hassaniya-b0681eeb (VITS, MMS-TTS fine-tuned on Hassaniya).
Uses data/generated/asr/recording_prompts.csv, so every clip has an exact transcript and intent.

Two uses:
  1. a command-domain ASR test (do community ASR models understand *commands*?):
        python3 scripts/synth_tts.py --n 150 && python3 scripts/bench_asr.py --tts
  2. synthetic training audio for Whisper (mix <=30% with real speech; research on Sudanese
     Arabic found TTS + self-training helps low-resource dialect ASR, arXiv 2601.06802).

Output (git-ignored): data/external/tts_commands/{wav/*.wav, manifest.jsonl}
Speed/pitch are randomly perturbed per clip for a little speaker variety.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "data" / "external" / "tts_commands"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlemineb/mms-tts-hassaniya-b0681eeb")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoTokenizer, VitsModel

    from hassaniya import strip_tashkeel

    tok = AutoTokenizer.from_pretrained(a.model)
    model = VitsModel.from_pretrained(a.model).eval()
    rows = list(csv.DictReader(open(ROOT / "data/generated/asr/recording_prompts.csv", encoding="utf-8")))
    rng = random.Random(a.seed)
    rng.shuffle(rows)
    (OUT / "wav").mkdir(parents=True, exist_ok=True)
    man = open(OUT / "manifest.jsonl", "w", encoding="utf-8")
    done = 0
    for r in rows:
        if done >= a.n:
            break
        text = strip_tashkeel(r["text"]).replace("،", " ").replace("؟", "")
        ids = tok(text, return_tensors="pt")
        if ids["input_ids"].shape[1] < 3:
            continue
        model.speaking_rate = rng.uniform(0.9, 1.15)
        with torch.no_grad():
            wav = model(**ids).waveform[0].numpy()
        sr = model.config.sampling_rate
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        wav = np.concatenate([np.zeros(4000), wav, np.zeros(4000)]).astype("float32")  # 0.25 s padding
        path = OUT / "wav" / f"{r['prompt_id']}.wav"
        sf.write(path, wav, 16000)
        man.write(json.dumps({"id": r["prompt_id"], "file": str(path.relative_to(OUT)), "text": text,
                              "intent": r["intent"], "duration_seconds": round(len(wav) / 16000, 2)}, ensure_ascii=False) + "\n")
        done += 1
    print(f"synthesized {done} clips -> {OUT}")


if __name__ == "__main__":
    main()

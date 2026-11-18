#!/usr/bin/env python3
"""Benchmark existing (community) Hassaniya ASR models on the MauriAI broadcast test set,
scored with MauriAI's own scorer, so the numbers are comparable with their published table.

    python3 scripts/bench_asr.py --n 100 --models all            # CPU is fine, ~1-2 h
    python3 scripts/bench_asr.py --n 408 --models hadramibegnoug/whisper-hassaniya-asr --device cuda

Outputs: data/benchmarks/asr/<model>.jsonl (predictions), data/benchmarks/asr/results.md
Audio/transcripts: CC-BY-NC-4.0 + third-party rights (see the dataset card); they are cached
in data/external/ and NOT committed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "external" / "mauriai_test"
OUT = ROOT / "data" / "benchmarks" / "asr"
BASE = "https://huggingface.co/datasets/MauriAI-lab/hassaniya-speech-test-v1/resolve/main/"

MODELS = [  # (repo, kind) - community fine-tunes first, slow medium models last
    ("hadramibegnoug/whisper-hassaniya-asr", "seq2seq"),
    ("tahermedsk/whisper-hassaniya", "seq2seq"),
    ("hadramibegnoug/mms-hassaniya-asr", "ctc"),
    ("abdourrahmane/mms-hassaniya-ctc", "ctc"),
    ("openai/whisper-small", "seq2seq"),                       # zero-shot baseline, same size as most community FTs
    ("DesertMindAI/ASR-hassaniya-whisper-medium-v1", "seq2seq"),
    ("HamidAmeer/hassaniya-whisper-medium", "seq2seq"),
]
PUBLISHED = ["mauriai_cohere_ft_37h", "zero_shot_whisper_large_v3", "zero_shot_qwen3_asr_1.7b",
             "zero_shot_qwen3_asr_0.6b", "zero_shot_cohere_transcribe_arabic"]


def fetch(name: str) -> Path:
    p = CACHE / name
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(BASE + name, timeout=120) as r:
            p.write_bytes(r.read())
    return p


def subset(n: int):
    rows = [json.loads(l) for l in fetch("test.jsonl").read_text(encoding="utf-8").splitlines()]
    if n >= len(rows):
        return rows
    by_vid = defaultdict(list)
    for r in rows:
        by_vid[r["video_id"]].append(r)
    out, i = [], 0
    while len(out) < n:  # round-robin over source videos => every video represented
        for v in sorted(by_vid):
            if i < len(by_vid[v]) and len(out) < n:
                out.append(by_vid[v][i])
        i += 1
    return out


def score_rows(rows, preds):
    sys.path.insert(0, str(ROOT / "scripts"))
    from hsn_score import score
    hyp = {p["id"]: p["text"] for p in preds}
    return score([(r["text"], hyp[r["id"]]) for r in rows if r["id"] in hyp], fetch("house_spelling.json"))


def run_model(repo, kind, rows, device):
    import soundfile as sf
    import torch
    from transformers import pipeline

    kw = {"model": repo, "device": 0 if device == "cuda" else -1, "torch_dtype": torch.float16 if device == "cuda" else torch.float32}
    pipe = pipeline("automatic-speech-recognition", **kw)
    gk = {"generate_kwargs": {"language": "ar", "task": "transcribe", "max_new_tokens": 160}} if kind == "seq2seq" else {}
    preds, t0, audio_s = [], time.time(), 0.0
    for r in rows:
        wav, sr = sf.read(r["file"] if Path(r["file"]).is_absolute() else fetch(r["file"]))
        audio_s += len(wav) / sr
        try:
            txt = pipe({"raw": wav.astype("float32"), "sampling_rate": sr}, **gk)["text"]
        except Exception as e:  # noqa: BLE001
            print("   err", r["id"], e)
            txt = ""
        preds.append({"id": r["id"], "text": txt.strip()})
    rtf = (time.time() - t0) / max(audio_s, 1)
    del pipe
    return preds, rtf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--models", nargs="*", default=["all"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tts", action="store_true", help="command-domain test on data/external/tts_commands (scripts/synth_tts.py)")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(max(1, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    if a.tts:
        return run_tts(a)
    rows = subset(a.n)
    fetch("house_spelling.json")
    results = []
    ids = {r["id"] for r in rows}
    for p in PUBLISHED:  # published predictions, restricted to the same subset
        try:
            preds = [json.loads(l) for l in fetch(f"predictions/{p}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        except Exception as e:  # noqa: BLE001
            print("skip", p, e)
            continue
        results.append((f"{p} (published predictions)", None, score_rows(rows, [x for x in preds if x["id"] in ids])))
        print(results[-1][0], results[-1][2], flush=True)
    todo = MODELS if a.models == ["all"] else [(m, k) for m, k in MODELS if m in a.models] or [(m, "seq2seq") for m in a.models]
    for repo, kind in todo:
        print(f"== {repo} ({kind})", flush=True)
        try:
            preds, rtf = run_model(repo, kind, rows, a.device)
        except Exception as e:  # noqa: BLE001
            print("   FAILED", e)
            results.append((repo, None, {}))
            continue
        pp = OUT / f"{repo.replace('/', '__')}.n{len(rows)}.jsonl"
        pp.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds), encoding="utf-8")
        s = score_rows(rows, preds)
        results.append((repo, rtf, s))
        print(f"   RTF={rtf:.2f}", s, flush=True)
        write_md(results, len(rows), a.device)
    write_md(results, len(rows), a.device)


def run_tts(a):
    """ASR on synthetic Hassaniya *commands*, then HA's hassil matcher on the transcript:
    end-to-end intent accuracy = would Home Assistant's fast path have done the right thing?"""
    import yaml
    from hassil import Intents, recognize
    tdir = ROOT / "data" / "external" / "tts_commands"
    rows = [json.loads(l) for l in open(tdir / "manifest.jsonl", encoding="utf-8")]
    for r in rows:
        r["file"] = str(tdir / r["file"])
    intents = Intents.from_dict(yaml.safe_load(open(ROOT / "homeassistant/custom_sentences/ar/hassaniya.yaml", encoding="utf-8")))

    def e2e(texts):
        ok = n = 0
        for r, t in zip(rows, texts):
            if not r["intent"].startswith("Hass"):
                continue
            n += 1
            res = recognize(t.replace("،", " ").replace("؟", " ").strip(), intents)
            ok += bool(res and res.intent.name == r["intent"])
        return 100 * ok / max(n, 1), n

    gold_acc, n_cmd = e2e([r["text"] for r in rows])
    lines = [f"# Command-domain ASR test: {len(rows)} synthetic Hassaniya commands\n",
             "Audio comes from `scripts/synth_tts.py`, which uses one synthetic speaker (the community VITS voice "
             "mlemineb/mms-tts-hassaniya), so these numbers are optimistic compared with real voices. "
             "**E2E** is the share of Hass* commands where Home Assistant's hassil matcher, run with our Hassaniya sentences "
             f"on the ASR transcript, recovers the right intent. With perfect transcripts it is {gold_acc:.0f}% (n={n_cmd}).\n",
             "| Model | Step 3 CER % | Step 3 WER % | E2E intent % | RTF |", "|---|---|---|---|---|"]
    todo = MODELS if a.models == ["all"] else [(m, k) for m, k in MODELS if m in a.models]
    for repo, kind in todo:
        print("==", repo, flush=True)
        try:
            preds, rtf = run_model(repo, kind, rows, a.device)
        except Exception as e:  # noqa: BLE001
            print("   FAILED", e)
            continue
        s = score_rows(rows, preds)
        acc, _ = e2e([p["text"] for p in preds])
        (OUT / f"tts__{repo.replace('/', '__')}.jsonl").write_text(
            "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds), encoding="utf-8")
        lines.append(f"| {repo} | {s['step3'][0]:.1f} | {s['step3'][1]:.1f} | {acc:.0f} | {rtf:.2f} |")
        print("  ", s, f"e2e={acc:.0f}%", flush=True)
        (OUT / "results_commands.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_md(results, n, device):
    lines = [f"# Hassaniya ASR benchmark: MauriAI Hassaniya Speech Test v1 ({n} clips"
             + (", round-robin subset over all 12 source videos" if n < 408 else ", full set") + ")\n",
             "Scored with `scripts/hsn_score.py`, a reimplementation of MauriAI's documented scorer. "
             "Step 3 is MauriAI's main orthography-aware score; Light is minimal normalization. "
             f"RTF is processing time divided by audio time on `{device}` (4 vCPU).\n",
             "| Model | Step 3 CER % | Step 3 WER % | Light CER % | Light WER % | RTF |", "|---|---|---|---|---|---|"]
    for name, rtf, s in sorted(results, key=lambda x: x[2].get("step3", (999, 999))[1]):
        f = lambda k, i: f"{s[k][i]:.1f}" if k in s else "failed"  # noqa: E731
        lines.append(f"| {name} | {f('step3', 0)} | {f('step3', 1)} | {f('light', 0)} | {f('light', 1)} | "
                     + (f"{rtf:.2f}" if rtf is not None else "–") + " |")
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

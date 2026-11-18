#!/usr/bin/env python3
"""Fine-tune Whisper for Hassaniya speech -> Arabic-script text (the STT half of the pipeline).

Data sources (any combination):
  * --hf MauriAI-lab/hassaniya-read-speech-v1   (~4.4 h read speech, CC-BY-NC-4.0, check license)
  * --local data/recordings/                    (your own recordings made with tools/recorder.html:
                                                  <prompt_id>.webm|wav + data/generated/asr/recording_prompts.csv)
Evaluation: MauriAI-lab/hassaniya-speech-test-v1 (broadcast speech, test split) and/or held-out local prompts.

    pip install -r training/requirements.txt
    python3 training/finetune_whisper.py --base openai/whisper-small \
        --hf MauriAI-lab/hassaniya-read-speech-v1 --local data/recordings --out runs/whisper-hsn
    # then for Home Assistant (wyoming-faster-whisper) convert to CTranslate2:
    ct2-transformers-converter --model runs/whisper-hsn --output_dir runs/whisper-hsn-ct2 \
        --copy_files tokenizer.json preprocessor_config.json --quantization int8

Text is normalized with scripts/hassaniya.normalize-like rules (tashkeel removed) so WER isn't
dominated by spelling variants. Check column names of the HF datasets with --inspect first.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def norm(t: str) -> str:
    from hassaniya import strip_tashkeel
    t = strip_tashkeel(t)
    t = re.sub("[أإآ]", "ا", t).replace("ى", "ي")
    t = re.sub(r"[؟?!.,،]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def load_local(folder: Path):
    from datasets import Audio, Dataset
    prompts = {r["prompt_id"]: r["text"] for r in csv.DictReader(
        open(ROOT / "data/generated/asr/recording_prompts.csv", encoding="utf-8"))}
    rows = []
    for f in sorted(folder.glob("**/*")):
        pid = f.stem.split("__")[0]
        if f.suffix.lower() in (".wav", ".webm", ".ogg", ".flac", ".mp3") and pid in prompts:
            rows.append({"audio": str(f), "text": prompts[pid]})
    print(f"local recordings: {len(rows)}")
    return Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=16000)) if rows else None


def load_hf(repo: str, split: str, text_col: str | None):
    from datasets import Audio, load_dataset
    ds = load_dataset(repo, split=split)
    col = text_col or next(c for c in ("text", "transcription", "sentence", "transcript") if c in ds.column_names)
    ds = ds.rename_column(col, "text") if col != "text" else ds
    return ds.select_columns(["audio", "text"]).cast_column("audio", Audio(sampling_rate=16000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="openai/whisper-small")
    ap.add_argument("--hf", nargs="*", default=[])
    ap.add_argument("--hf-split", default="train")
    ap.add_argument("--hf-text-col")
    ap.add_argument("--eval-hf", default="MauriAI-lab/hassaniya-speech-test-v1")
    ap.add_argument("--local")
    ap.add_argument("--out", default="runs/whisper-hsn")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()

    import evaluate
    import torch
    from datasets import concatenate_datasets
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration,
                              WhisperProcessor)

    if a.inspect:
        from datasets import load_dataset
        for r in a.hf + [a.eval_hf]:
            print(r, load_dataset(r, streaming=True))
        return

    parts = [load_hf(r, a.hf_split, a.hf_text_col) for r in a.hf]
    if a.local:
        loc = load_local(Path(a.local))
        if loc is not None:
            parts.append(loc)
    train = concatenate_datasets(parts).shuffle(seed=1)
    split = train.train_test_split(test_size=0.05, seed=1)
    train, dev = split["train"], split["test"]
    try:
        dev = load_hf(a.eval_hf, "test", None)
    except Exception as e:  # noqa: BLE001
        print("eval set not available, using held-out train split:", e)

    proc = WhisperProcessor.from_pretrained(a.base, language="arabic", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(a.base)
    model.generation_config.language = "arabic"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    def prep(b):
        au = b["audio"]
        b["input_features"] = proc.feature_extractor(au["array"], sampling_rate=16000).input_features[0]
        b["labels"] = proc.tokenizer(norm(b["text"])).input_ids
        return b

    train = train.map(prep, remove_columns=train.column_names, num_proc=4)
    dev = dev.map(prep, remove_columns=dev.column_names, num_proc=4)

    def collate(feats):
        x = proc.feature_extractor.pad([{"input_features": f["input_features"]} for f in feats], return_tensors="pt")
        y = proc.tokenizer.pad([{"input_ids": f["labels"]} for f in feats], return_tensors="pt")
        lab = y["input_ids"].masked_fill(y.attention_mask.ne(1), -100)
        if (lab[:, 0] == model.config.decoder_start_token_id).all():
            lab = lab[:, 1:]
        x["labels"] = lab
        return x

    wer = evaluate.load("wer")

    def metrics(p):
        pred = proc.batch_decode(p.predictions, skip_special_tokens=True)
        lab = p.label_ids
        lab[lab == -100] = proc.tokenizer.pad_token_id
        ref = proc.batch_decode(lab, skip_special_tokens=True)
        return {"wer": wer.compute(predictions=[norm(x) for x in pred], references=[norm(x) for x in ref])}

    args = Seq2SeqTrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.bs, gradient_accumulation_steps=1, learning_rate=a.lr,
        warmup_steps=300, max_steps=a.steps, gradient_checkpointing=True, fp16=torch.cuda.is_available(),
        eval_strategy="steps", eval_steps=500, save_steps=500, predict_with_generate=True,
        generation_max_length=128, logging_steps=25, report_to="none", load_best_model_at_end=True,
        metric_for_best_model="wer", greater_is_better=False)
    Seq2SeqTrainer(model=model, args=args, train_dataset=train, eval_dataset=dev, data_collator=collate,
                   compute_metrics=metrics, processing_class=proc.feature_extractor).train()
    model.save_pretrained(a.out)
    proc.save_pretrained(a.out)


if __name__ == "__main__":
    main()

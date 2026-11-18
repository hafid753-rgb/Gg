#!/usr/bin/env python3
"""LoRA fine-tune a small local LLM on the Hassaniya Home-Assistant tool-calling set.

Default base: Qwen2.5-3B-Instruct (good Arabic, native tool-calling chat template, runs
on CPU/8GB GPU after Q4 quantization). Alternatives: Qwen2.5-7B-Instruct, Llama-3.2-3B-Instruct,
gemma-3-4b-it (adjust --target-modules if needed).

    pip install -r training/requirements.txt
    python3 training/finetune_llm_lora.py --base Qwen/Qwen2.5-3B-Instruct --out runs/dar-3b
    python3 training/finetune_llm_lora.py --merge runs/dar-3b   # -> runs/dar-3b/merged
Then convert to GGUF + Ollama: see training/README.md.

Loss is applied to assistant turns only (tool calls + Hassaniya replies), never to the
system context or user text.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "data" / "generated" / "sft"
TOOLS = json.loads((ROOT / "data" / "schema" / "ha_tools.json").read_text(encoding="utf-8"))


def load_split(name: str, tok, max_len: int, aux_ratio: float = 0.0):
    from datasets import Dataset

    rows = []
    lines = list(open(SFT / f"{name}.jsonl", encoding="utf-8"))
    aux = SFT.parent / "sft_aux" / f"{name}.jsonl"
    if aux_ratio > 0 and aux.exists():  # real-Hassaniya auxiliary tasks (scripts/build_aux_sft.py)
        import random
        extra = list(open(aux, encoding="utf-8"))
        k = min(len(extra), int(len(lines) * aux_ratio / (1 - aux_ratio)))
        lines += random.Random(0).sample(extra, k)
        print(f"{name}: + {k} auxiliary real-Hassaniya examples")
    for line in lines:
        r = json.loads(line)
        msgs = r["messages"]
        # Qwen/HF chat templates expect tool-call arguments as dicts (they are) and tool results as strings.
        ids, labels = [], []
        # Build incrementally so we can mask everything that isn't produced by the assistant.
        prev = []
        for i, m in enumerate(msgs):
            prev.append(m)
            text = tok.apply_chat_template(prev, tools=TOOLS if "intent" in r else None, tokenize=False,
                                           add_generation_prompt=False)
            full = tok(text, add_special_tokens=False)["input_ids"]
            new = full[len(ids):]
            ids = full
            labels += new if m["role"] == "assistant" else [-100] * len(new)
        if len(ids) > max_len:
            continue
        rows.append({"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)})
    return Dataset.from_list(rows)


def train(a):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              Trainer, TrainingArguments)

    tok = AutoTokenizer.from_pretrained(a.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    kw = {"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
    if a.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                                       bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(a.base, **kw)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0.05,
                                             target_modules=a.target_modules.split(","), task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    train_ds = load_split("train", tok, a.max_len, a.aux_ratio)
    dev_ds = load_split("dev", tok, a.max_len)
    print(f"train={len(train_ds)} dev={len(dev_ds)}")
    args = TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs,
        gradient_accumulation_steps=a.accum, learning_rate=a.lr, num_train_epochs=a.epochs,
        lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=20, eval_strategy="steps",
        eval_steps=200, save_steps=200, save_total_limit=3, bf16=torch.cuda.is_available(),
        report_to="none", group_by_length=True, load_best_model_at_end=True)
    Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds,
            data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)).train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)


def merge(run_dir: str, base: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16)
    m = PeftModel.from_pretrained(m, run_dir).merge_and_unload()
    out = Path(run_dir) / "merged"
    m.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(run_dir).save_pretrained(out)
    print("merged ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out", default="runs/dar-3b")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=4096)  # 23 tool schemas ~2.5k tokens; median example ~3.1k
    ap.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA (needs bitsandbytes + CUDA)")
    ap.add_argument("--aux-ratio", type=float, default=0.15, help="share of real-Hassaniya auxiliary examples")
    ap.add_argument("--merge", metavar="RUN_DIR")
    a = ap.parse_args()
    merge(a.merge, a.base) if a.merge else train(a)

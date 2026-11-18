#!/usr/bin/env python3
"""Evaluate a local model served by Ollama (or any OpenAI-compatible /v1 endpoint) on the
held-out SFT conversations: does it call the right tools with the right arguments, and does
it answer in Arabic script (Hassaniya) when no tool is needed?

    ollama create dar -f training/Modelfile
    python3 training/eval_llm.py --model dar --n 300
    python3 training/eval_llm.py --model qwen2.5:3b --n 300      # baseline before fine-tuning
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARABIC = re.compile(r"[؀-ۿ]")
HASSANIYA_MARKERS = ["دُرك", "ذرك", "گ", "ياسر", "حاضر", "شاعل", "طافي", "من گبيل", "ماهو", "اشطاري", "گاع"]


def chat(host, model, messages, tools):
    body = json.dumps({"model": model, "messages": messages, "tools": tools, "stream": False,
                       "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(f"{host}/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["message"]


def norm_args(a):
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except json.JSONDecodeError:
            return {}
    return {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in (a or {}).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dar")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    tools = json.loads((ROOT / "data/schema/ha_tools.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in open(ROOT / f"data/generated/sft/{a.split}.jsonl", encoding="utf-8")][: a.n]
    c = Counter()
    for r in rows:
        msgs = r["messages"]
        # evaluate the first assistant decision after the first user turn
        prompt = msgs[:2]
        gold = msgs[2]
        out = chat(a.host, a.model, prompt, tools)
        gold_calls = [(t["function"]["name"], norm_args(t["function"]["arguments"])) for t in gold.get("tool_calls") or []]
        pred_calls = [(t["function"]["name"], norm_args(t["function"]["arguments"])) for t in out.get("tool_calls") or []]
        c["n"] += 1
        if gold_calls:
            c["need_tool"] += 1
            c["tool_names_ok"] += sorted(n for n, _ in pred_calls) == sorted(n for n, _ in gold_calls)
            c["exact"] += sorted(json.dumps(x, sort_keys=True) for x in pred_calls) == sorted(json.dumps(x, sort_keys=True) for x in gold_calls)
        else:
            c["no_tool"] += 1
            txt = out.get("content", "")
            c["no_tool_ok"] += not pred_calls
            c["arabic_reply"] += bool(ARABIC.search(txt))
            c["hassaniya_marker"] += any(m in txt for m in HASSANIYA_MARKERS)
    p = lambda k, d: f"{c[k] / max(c[d], 1):.1%}"  # noqa: E731
    print(json.dumps({
        "model": a.model, "n": c["n"],
        "tool_name_accuracy": p("tool_names_ok", "need_tool"),
        "tool_call_exact_match": p("exact", "need_tool"),
        "correctly_no_tool": p("no_tool_ok", "no_tool"),
        "arabic_script_reply": p("arabic_reply", "no_tool"),
        "hassaniya_marker_rate": p("hassaniya_marker", "no_tool"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

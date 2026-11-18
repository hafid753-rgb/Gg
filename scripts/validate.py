#!/usr/bin/env python3
"""Validate the generated dataset.

1. SFT structural checks: tool names/args match data/schema/ha_tools.json, every tool call is
   answered, every `name` argument exists in the system context, final reply is non-empty
   Arabic script.
2. Home-Assistant fast path: run hassil (HA's own matcher) with
   homeassistant/custom_sentences/ar/hassaniya.yaml over held-out clean NLU utterances and
   report coverage + intent/slot accuracy.

    python3 scripts/validate.py [--split test] [--show 15]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "data" / "generated"
ARABIC = re.compile(r"[؀-ۿ]")


def check_sft(split: str) -> int:
    tools = json.loads((ROOT / "data/schema/ha_tools.json").read_text(encoding="utf-8"))
    schema = {t["function"]["name"]: set(t["function"]["parameters"]["properties"]) for t in tools}
    errors = Counter()
    n = 0
    for line in open(GEN / "sft" / f"{split}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        n += 1
        msgs = r["messages"]
        ctx = msgs[0]["content"]
        if msgs[0]["role"] != "system" or msgs[1]["role"] != "user":
            errors["bad_header"] += 1
        pending = []
        for m in msgs:
            for c in m.get("tool_calls", []) or []:
                f = c["function"]
                if f["name"] not in schema:
                    errors[f"unknown_tool:{f['name']}"] += 1
                bad = set(f["arguments"]) - schema.get(f["name"], set())
                if bad:
                    errors[f"bad_args:{f['name']}:{sorted(bad)}"] += 1
                nm = f["arguments"].get("name")
                if nm and f"names: {nm}\n" not in ctx:
                    errors["name_not_in_context"] += 1
                pending.append(f["name"])
            if m["role"] == "tool":
                if not pending or pending.pop(0) != m["name"]:
                    errors["tool_result_mismatch"] += 1
        if pending:
            errors["unanswered_tool_call"] += 1
        last = msgs[-1]
        if last["role"] != "assistant" or not last["content"].strip() or not ARABIC.search(last["content"]):
            errors["bad_final_reply"] += 1
    print(f"[sft/{split}] {n} conversations, errors: {dict(errors) or 'none'}")
    return sum(errors.values())


def check_hassil(split: str, show: int) -> float:
    from hassil import Intents, recognize

    doc = yaml.safe_load(open(ROOT / "homeassistant/custom_sentences/ar/hassaniya.yaml", encoding="utf-8"))
    intents = Intents.from_dict(doc)
    rows = [json.loads(l) for l in open(GEN / "nlu" / f"{split}.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["fast_path"]]
    per = defaultdict(lambda: Counter())
    misses = []
    for r in rows:
        text = re.sub(r"[،؟?!.,]", " ", r["text"])
        text = re.sub(r"\s+", " ", text).strip()
        res = recognize(text, intents)
        c = per[r["intent"]]
        c["n"] += 1
        if res is None:
            misses.append(("NO_MATCH", r["intent"], r["slots"], r["text"]))
            continue
        c["matched"] += 1
        got = {k: v.value for k, v in res.entities.items()}
        if res.intent.name != r["intent"]:
            misses.append((f"WRONG:{res.intent.name}", r["intent"], r["slots"], r["text"]))
            continue
        def same(a, b):
            try:
                return float(a) == float(b)
            except (TypeError, ValueError):
                return str(a) == str(b)
        ok = all(same(got.get(k), v) for k, v in r["slots"].items() if k in got or k in ("area", "domain"))
        if ok:
            c["correct"] += 1
        else:
            misses.append((f"SLOTS:{got}", r["intent"], r["slots"], r["text"]))
    tot = sum(c["n"] for c in per.values())
    matched = sum(c["matched"] for c in per.values())
    correct = sum(c["correct"] for c in per.values())
    print(f"[hassil/{split}] fast-path utterances: {tot}  matched: {matched} ({matched / max(tot, 1):.1%})  "
          f"fully correct: {correct} ({correct / max(tot, 1):.1%})")
    for intent, c in sorted(per.items(), key=lambda x: -x[1]["n"]):
        print(f"   {intent:28s} n={c['n']:5d}  match={c['matched'] / c['n']:.0%}  correct={c['correct'] / c['n']:.0%}")
    for m in misses[:show]:
        print("   MISS", m)
    return correct / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--show", type=int, default=15)
    ap.add_argument("--min-hassil", type=float, default=0.0)
    a = ap.parse_args()
    errs = sum(check_sft(s) for s in ("train", "dev", "test"))
    acc = check_hassil(a.split, a.show)
    if errs or acc < a.min_hassil:
        sys.exit(1)


if __name__ == "__main__":
    main()

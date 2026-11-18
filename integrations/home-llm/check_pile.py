#!/usr/bin/env python3
"""Static check of the Hassaniya pile against home-llm's loader rules (data/utils.py,
generate_data.py as of 2026-11), without importing or running home-llm code.

    python3 check_pile.py piles/hassaniya [path/to/home-llm/data/piles/english]
"""
import csv
import re
import sys
from pathlib import Path

pile = Path(sys.argv[1])
ref = Path(sys.argv[2]) if len(sys.argv) > 2 else None
TYPES = {"light", "switch", "fan", "garage_door", "blinds", "lock", "media_player", "climate", "vacuum", "timer", "todo"}
VAR = re.compile("<(.*?)>")
errors = []


def rows(name):
    with open(pile / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


REQUIRED = ["pile_of_and_words.csv", "pile_of_durations.csv", "pile_of_todo_items.txt", "pile_of_device_names.csv",
            "pile_of_templated_actions.csv", "pile_of_specific_actions.csv", "pile_of_responses.csv",
            "pile_of_status_requests.csv", "pile_of_system_prompts.csv", "pile_of_refusals.csv", "pile_of_failed_tool_calls.csv"]
for n in REQUIRED:
    if not (pile / n).exists():
        errors.append(f"missing {n}")
    elif ref and n.endswith(".csv") and n != "pile_of_and_words.csv":
        h1 = open(pile / n, encoding="utf-8").readline().strip()
        h2 = open(ref / n, encoding="utf-8").readline().strip()
        if h1 != h2:
            errors.append(f"{n}: header {h1!r} != english {h2!r}")

devs = rows("pile_of_device_names.csv")
by_type = {}
for d in devs:
    t = d["device_name"].split(".")[0]
    if t not in TYPES:
        errors.append(f"device type {t} unsupported")
    by_type.setdefault(t, []).append(d)

resp = rows("pile_of_responses.csv")
personas = {r["persona"] for r in rows("pile_of_system_prompts.csv")}
avail = {(r["service"], r["persona"], int(r["short"]), ",".join(sorted(set(VAR.findall(r["response_starting"])) - {"device_name"})))
         for r in resp}
for t in rows("pile_of_templated_actions.csv"):
    int(t["multiplier"])
    types, svcs = t["device_type"].split("|"), t["service"].split("|")
    short = len(types) > 1
    req = ",".join(sorted({v for v in VAR.findall(t["phrase"]) if "device_name" not in v}))
    for dt, sv in zip(types, svcs):
        if dt not in by_type:
            errors.append(f"template uses {dt} but no device names of that type")
        for p in personas:
            if (f"{dt}.{sv}", p, int(short), req) not in avail:
                errors.append(f"no response for {dt}.{sv} persona={p} short={short} vars=[{req}] <- {t['phrase']}")
for a in rows("pile_of_specific_actions.csv"):
    if not any((a["service_name"], p, 0, "") in avail for p in personas):
        errors.append(f"no response for specific action {a['service_name']}")
for s in rows("pile_of_status_requests.csv"):
    if s["device_type"] not in by_type:
        errors.append(f"status request for {s['device_type']} has no devices")

print(f"{pile}: {len(devs)} devices, {len(rows('pile_of_templated_actions.csv'))} templates, {len(resp)} responses, "
      f"{len(rows('pile_of_status_requests.csv'))} status requests -> {len(errors)} problems")
for e in sorted(set(errors))[:40]:
    print("  -", e)
sys.exit(1 if errors else 0)

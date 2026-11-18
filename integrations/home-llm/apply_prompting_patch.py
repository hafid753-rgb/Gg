#!/usr/bin/env python3
"""Add the `hassaniya` entries to home-llm's data/prompting.py (idempotent).

    python3 apply_prompting_patch.py /path/to/home-llm/data/prompting.py
"""
import re
import sys
from pathlib import Path

ENTRIES = {
    "CURRENT_DATE_PROMPT": "الساعة والتاريخ ظرك",
    "DEVICES_PROMPT": "الآلات",
    "SERVICES_PROMPT": "الخدمات",
    "BABEL_LOCALE": "ar_MR",
    "BABEL_FORMAT": "H:mm EEEE d MMMM yyyy",
    "USER_INSTRUCTION_PROMPT": "طلب صاحب الدار",
}

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
for name, value in ENTRIES.items():
    m = re.search(rf"{name}\s*=\s*\{{(.*?)\n\}}", s, re.S)
    if not m:
        sys.exit(f"could not find {name} in {p}")
    if '"hassaniya"' in m.group(1):
        continue
    body = m.group(1).rstrip()
    sep = "" if body.endswith(",") else ","
    s = s[:m.start(1)] + body + f'{sep}\n    "hassaniya": "{value}"' + s[m.end(1):]
p.write_text(s, encoding="utf-8")
print("patched", p)

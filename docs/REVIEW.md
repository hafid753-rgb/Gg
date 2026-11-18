# Native-speaker review protocol

The whole dataset is generated from `data/lexicon/lexicon.yaml`, and the assistant's reply templates live in `scripts/generate.py`, in `reply_*`, `SOCIAL`, `OOS`, and the `sft_*` builders. **Reviewing those two places reviews all 45k samples.**

## Pass 1: Lexicon (≈2 hours, 2+ reviewers from different regions)
For every entry, mark one of:
- ✅ natural Hassaniya
- 🔁 understood but not how we say it (give the natural form)
- ❌ wrong, Moroccan, or MSA-only

Priorities:
1. Entries tagged `conf: low` or `conf: med`.
2. Verbs, especially *close* (سدّ / اغلگ / بلّع), *cancel*, *clean*, and the polite `req` forms.
3. Room names (بيت الرگاد vs بيت النعاس vs الشومبر) and the French loans people actually use in Nouakchott, Nouadhibou, Atar and Kiffa.
4. Spelling conventions: گ vs ق vs ڭ, ذ vs د, how to write French loans.

Add regional variants instead of replacing (the list is the variation). Put the most common form **first**, because the first form becomes the canonical form used in replies and entity names.

## Pass 2: Replies (≈1 hour)
Run:
```bash
python3 - <<'EOF'
import json, random
rows = [json.loads(l) for l in open('data/generated/sft/dev.jsonl')]
for r in random.sample(rows, 50):
    m = r['messages']; print('USER:', m[1]['content']); print('DAR :', m[-1]['content']); print()
EOF
```
For each reply, check that:
- it sounds like a Mauritanian talking, not a news anchor
- gender agreement is right (الضو شاعل / اللامبة شاعلة)
- it is polite enough
- it is not too long

## Pass 3: Sample audit (≈30 min)
Sample 100 NLU rows. For each, judge whether a real person would say it, and whether the intent and slots are correct. Track the error rate in this file over versions.

## Recording (ASR)
- Use `tools/recorder.html` with `data/generated/asr/recording_prompts.csv`.
- Aim for **10+ speakers** with a gender balance and several regions. Include older speakers and kids if they will use the system.
- Have each speaker record 200–400 prompts at home, on the microphone the satellite will use.
- Get written consent. Use speaker IDs, not names. Keep the audio off public repositories unless every speaker agreed to it.
- Also collect **free speech**: ask each speaker to say 20 commands in their own words. Transcribe those by hand. They are the most valuable data you will have.

## Review log
| Date | Reviewer (id) | Region | Pass | Changes |
|---|---|---|---|---|
| 2026-11-18 | repo owner | – | 1 | "on" state: گادي / گادية preferred over شاعل (both correct). Replies now use گادي; شاعل kept as an understood user variant. |
| 2026-11-18 | repo owner | – | 1 | "off" state: مطفي / مطفية preferred, طافي understood. AC: الكليم, الكليماتيزور, المكيّف confirmed. Bedroom: بيت الرگاد, البيت (removed بيت النعاس, الشومبر). حگل confirmed very natural before or after a command (now in ~14% of commands). |
| 2026-11-18 | repo owner | – | 2 | Switch on: only اگدي is natural (اشعل/شعّل/حلّ kept as understood-only variants). Acknowledgement: ايوه (replaces اهيه/زين/تمّ). Close curtains/windows: اگفل (سكّر/سدّ understood-only). Wake word: keep يا دار. |
| 2026-11-18 | repo owner | – | 3 | Water heater: الشوفو only (removed سخّان الما). Fan: المروحة, الفانتيلو, الفانتيلاتير, الفنتيلاتور confirmed. Kitchen: الكوزينة, المطبخ; living room: الصالون. Curtains: الريدو, الستارة, الريدوات confirmed. |
| | | | | |

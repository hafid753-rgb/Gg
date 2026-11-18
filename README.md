# Gg: a Hassaniya voice dataset for Home Assistant

This repository holds a dataset and a full local toolchain. The goal is a smart home that understands spoken **Hassaniya** (Mauritanian Arabic, ISO 639‑3 `mey`) and answers back in Hassaniya. Everything runs on your own hardware, with no cloud.

```
🎙 voice ─► Whisper (fine-tuned on Hassaniya) ─► HA Assist
                                                  ├─ fast path: Hassaniya custom sentences (hassil)  ~ms, no LLM
                                                  └─ fallback: local LLM (LoRA on this dataset, via Ollama)
                                                        └─ calls HA tools, replies dynamically in Hassaniya ─► Piper TTS 🔊
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full pipeline and hardware notes.

## Built on existing work
This repository doesn't start from scratch. [`docs/RESEARCH.md`](docs/RESEARCH.md) surveys everything public for Hassaniya: 11 datasets, more than 20 models, and the related GitHub projects. Here is what we reuse:

- **Real Hassaniya text → lexicon evidence.** `scripts/ingest_external.py` collects 16k sentences from 11 community sources. `scripts/lexicon_evidence.py` counts every lexicon form in them, which corrected the vocabulary to forms Mauritanians actually write: **ظرك** (now), **الظو** (light), **اگدي** (switch on), **اگفل** (close/lock), the **حگل** softener, and **اهيه** (yes). The 5.7k permissively licensed sentences are committed in `data/corpus/` with attribution.
- **MauriAI's Hassaniya Speech Test** is the ASR benchmark. `scripts/hsn_score.py` reimplements their scorer and reproduces their published table exactly. `scripts/bench_asr.py` scores the community Hassaniya ASR models, which have no published accuracy; results are in [`data/benchmarks/asr/results.md`](data/benchmarks/asr/results.md).
- **The community Hassaniya TTS** (`mlemineb/mms-tts-hassaniya`) was tested for synthetic command audio (`scripts/synth_tts.py`). Result: three good ASR models get 0% end-to-end on it, so the voice isn't intelligible enough. **Don't train on it.** See [`results_commands.md`](data/benchmarks/asr/results_commands.md). The end-to-end harness (speech → ASR → hassil intent) is ready for real recordings or a better TTS.
- **acon96/home-llm:** a Hassaniya language pack for their generator and integration, in `integrations/home-llm/`.
- **home-assistant/intents:** our Hassaniya sentences extend HA's official Arabic (MSA) set and follow its intent names.
- **hassan-IA/dah and amthal (CC-BY):** an auxiliary SFT set from real Hassaniya (translation, Arabizi→Arabic script, proverbs) keeps the LLM's replies authentic. See `scripts/build_aux_sft.py`.

## What's inside

| Path | What | Size (seed 7) |
|---|---|---|
| `data/lexicon/lexicon.yaml` | **Source of truth.** Hassaniya verbs (m/f/pl/polite/"I want" forms), devices (incl. French loans: *klim, ventilo, chauffe-eau, rideau, pompe*), rooms (*ṣālūn, bayt er-rgād, kūzīna, khayma*…), numbers (Hassaniya + French), colors, units, function words (*durk, gāʿ, yāser, gaddāsh, wakt-āsh*…). Every entry has a confidence tag. | ~400 entries |
| `data/generated/nlu/*.jsonl` | Utterance → HA intent + slots. 70% clean, 22% ASR-noised (گ/ق/ك, ة/ه, glued `فال…`, tashkeel loss), 8% Arabizi (`ach3el ed-daw fi es-salon`). | 30,000 (26.9k/1.6k/1.5k) |
| `data/generated/sft/*.jsonl` | Tool-calling chat conversations. Each one is grounded in a randomly generated Mauritanian home (areas, devices, live states), using HA's own LLM-API prompt layout. | 15,000 (13.6k/0.7k/0.65k) |
| `data/generated/asr/recording_prompts.csv` | 2,000 diverse prompts for native speakers to record. | 2,000 |
| `data/schema/ha_tools.json` | The 20 HA intent tools (HassTurnOn, HassLightSet, HassClimateSetTemperature, HassStartTimer…). | |
| `homeassistant/custom_sentences/ar/hassaniya.yaml` | Generated HA custom sentences for the zero-LLM fast path. Covers 23 intents with 202 templates. | |
| `scripts/` | `generate.py`, `build_ha_sentences.py`, `validate.py`, `download_external.py`, `hassaniya.py` (normalizer, Arabizi, Hassaniya clock/number formatting). | |
| `training/` | LoRA SFT (`finetune_llm_lora.py`), Whisper fine-tune (`finetune_whisper.py`), Ollama eval (`eval_llm.py`), `Modelfile`. | |
| `tools/recorder.html` | Offline browser recorder for collecting Hassaniya speech against the prompt list. | |

### What the SFT data teaches (the "dynamic" part)

The model learns more than a fixed mapping from phrase to action. It learns to:

- **Use state.** It answers *"الظو گادي من گبيل"* when a light is already on instead of calling a tool. For *"زيد الظو شوي"* it reads the current brightness (40%), raises it to 60%, and says so.
- **Disambiguate.** It asks *"أي كليم؟ الّي ف الصالون ولّا بيت الرگاد؟"* and then acts on the answer across two turns.
- **Refuse honestly.** It says *"ما لگيت حتى كليم ف الكوزينة"* when a device doesn't exist, and suggests ones that do.
- **Act on indirect requests.**
  - *"الدار سخونة ياسر"* turns the AC on, or lowers its setpoint if it is already on. A home with no AC gets the fan.
  - *"لاهي نرگد"* turns off all lights and the TV, then locks the door.
  - *"انا خارج"* turns off the lights and AC and locks the door.
  - *"نبغي نغتسل"* turns on the chauffe-eau, and *"الما ما ينزل"* turns on the pump.
- **Handle several commands at once,** such as *"طفّي الظو ف الصالون واگدي الكليم ف بيت الرگاد"*.
- **Answer questions** about temperature, state, time, date and weather from tool results, in Hassaniya. Times use the local clock style (*"اربعة الّا ربع متاع لعشيّة"*), and weather answers add local advice (sandstorm *عجاج* → close the windows).
- **Handle social turns and out-of-scope requests** naturally (*اشطاري*, *شكرا*, *"اعدل لي أتاي"*).

## Quick start

```bash
pip install pyyaml hassil
make all            # regenerate everything from the lexicon + validate
```

The validation run checks two things:
- **Structure:** every tool call matches the schema, every tool result is answered, and every entity name exists in the context. Current result: 0 errors.
- **Fast path:** HA's own matcher (`hassil`) runs over held-out clean commands. Current result: 99.1% of 995 test utterances are parsed with the correct intent and slots.

Then:
1. **Home Assistant fast path:** copy `homeassistant/custom_sentences/ar/hassaniya.yaml` into `<config>/custom_sentences/ar/`. Full steps are in [`homeassistant/README.md`](homeassistant/README.md).
2. **LLM:** `python3 training/finetune_llm_lora.py`, then merge, convert to GGUF, and `ollama create dar`. Finally run `python3 training/eval_llm.py --model dar` and compare against the base model.
3. **Speech:** record prompts with `tools/recorder.html` and run `scripts/download_external.py` to fetch the public Hassaniya speech data. Then run `training/finetune_whisper.py` and convert the result to CTranslate2 for `wyoming-faster-whisper`.

## External Hassaniya resources (the main ones; full list in docs/RESEARCH.md)

| Resource | Type | Notes |
|---|---|---|
| [`MauriAI-lab/hassaniya-read-speech-v1`](https://huggingface.co/datasets/MauriAI-lab/hassaniya-read-speech-v1) | ASR train | ~4.4 h of read speech: 900 clips, 5 speakers. **CC‑BY‑NC‑4.0** |
| [`MauriAI-lab/hassaniya-speech-test-v1`](https://huggingface.co/datasets/MauriAI-lab/hassaniya-speech-test-v1) | ASR eval | ~1 h of broadcast speech, 408 clips, 16 kHz FLAC, test split only |
| [`hassan-IA/dah`](https://huggingface.co/datasets/hassan-IA/dah) | Text | Hassaniya↔English sentences in Arabic script and Arabizi, adapted from Tatoeba |
| Peace Corps Mauritania *Hassaniya Communication & Culture* / *Grammar Handbook* | Lexicon | 30 lessons with an English‑Hassaniya glossary. Good for lexicon review |
| *Spoken Hassaniya Arabic* (Dunwoody Press, 2008) | Lexicon | Transcribed audio with a glossary |
| C. Taine‑Cheikh, "Ḥassāniyya Arabic" (Language Science Press) | Grammar | Reference description of the dialect |

`scripts/ingest_external.py` pulls the text from all of them. `scripts/bench_asr.py` pulls the MauriAI test audio, and `scripts/download_external.py` pulls the speech sets for training. Non-commercial data is cached in the git-ignored `data/external/` and is never committed.

## ⚠️ Honest status and what's needed next

- **The lexicon still needs native-speaker review.** It is now checked against 16k real sentences: 193 of 374 forms are attested, and [`docs/LEXICON_EVIDENCE.md`](docs/LEXICON_EVIDENCE.md) lists every form with its hit count and suggested look-alike spellings. The unattested forms are mostly smart-home words that the corpus never mentions. They may be fine, but someone has to confirm them. Because every sentence is generated from `lexicon.yaml`, fixing a word there fixes all 45k+ samples on the next `make data`. The review protocol is in [`docs/REVIEW.md`](docs/REVIEW.md).
- **The data is synthetic.** It is template-generated, so it has the coverage of a grammar, not the messiness of real speech. The fix is to record the 2,000 prompts with several speakers from different regions and add real transcribed commands to `data/`.
- **Training hasn't been run (no GPU here).** The LoRA data pipeline was checked with the real Qwen2.5 tokenizer: label masking is correct, prefix tokenization is consistent, and the max length was raised to 4096 because the 23 tool schemas take about 2.5k tokens. The Whisper fine-tune script has not been run.
- **Licenses:** this repository's generated data is yours. Several external sets are non-commercial.

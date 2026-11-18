# Architecture: a fully local Hassaniya voice assistant

The assistant runs in five stages:

1. **Wake word.** openWakeWord or microWakeWord, on the satellite (ESP32‑S3, Voice PE, or a Raspberry Pi). Train a custom word such as *"يا دار"*.
2. **Speech to text.** `wyoming-faster-whisper`, loaded with our fine-tuned Whisper converted to CTranslate2. Pipeline language: `ar`.
3. **Home Assistant Assist** handles the transcript in one of two ways:
   - **Fast path:** `custom_sentences/ar/hassaniya.yaml` (hassil) matches the command in about 5 ms with no LLM. It handles on/off, setpoints, timers, time, weather and so on.
   - **Fallback:** the Ollama conversation agent runs model `dar`, a LoRA fine-tune on `data/generated/sft`. HA's LLM API gives it the exposed entities plus the Hass* tools, and the "prefer handling commands locally" option keeps the fast path first.
4. **Reply.** The LLM calls the tools and then writes a Hassaniya reply from the tool results and the entity states.
5. **Text to speech.** Piper, using an Arabic voice (`ar_JO-kareem`), or a Hassaniya voice you fine-tune from the recorded prompts.

## Why two paths
- Most commands are short and formulaic, so hassil handles them instantly and deterministically. It is about 99% accurate on held-out clean generated commands, and it can run on a Pi.
- Indirect, relative, stateful, multi-command and conversational requests need the LLM:
  - indirect: *"الدار سخونة"*
  - relative: *"زيد الضو شوي"*
  - stateful: already on, ambiguous room
  - multi-command and conversational turns
- HA's *"Prefer handling commands locally"* setting gives exactly this cascade.

## Model choices (all local)
| Stage | Recommended | Small hardware | Notes |
|---|---|---|---|
| STT | `whisper-large-v3-turbo` fine-tuned | `whisper-small` fine-tuned | Stock Whisper Arabic is MSA/Gulf-biased. Fine-tune on MauriAI read speech plus your recordings, and evaluate WER on MauriAI broadcast test. |
| LLM | Qwen2.5‑7B‑Instruct + LoRA, Q4_K_M (~5 GB) | Qwen2.5‑3B‑Instruct + LoRA, Q4_K_M (~2 GB) | Both have native tool-calling templates and decent Arabic. 3B works well with an 8 GB GPU or a fast CPU. |
| TTS | Piper fine-tuned on Hassaniya speaker recordings | Piper `ar_JO-kareem` | An MSA-accent voice is intelligible but not dialectal. |

Latency targets on an N100 mini PC or a used GPU:
- Whisper‑small int8: ~0.5 s.
- Fast path: ~0.01 s.
- 3B LLM: ~1–2 s on GPU, 3–6 s on CPU.

## The SFT prompt format
The format mirrors what HA's Ollama integration sends:
- A system message containing the instructions plus a `Static Context` YAML of entities, each with `names`, `domain`, `state`, `areas` and `attributes`.
- Tools in OpenAI/Ollama function format, from `data/schema/ha_tools.json`.
- Assistant `tool_calls` whose `arguments` are JSON objects, followed by `tool` messages in HA's `action_done` / `query_answer` result shape, and then the final Hassaniya reply.

Newer HA versions may move live states behind a `GetLiveContext` tool instead of the prompt. If yours does, adapt `home_context()` in `scripts/generate.py` and regenerate. The generator is the single place to change the format.

## Data flow
```
lexicon.yaml ──► generate.py ──► nlu/, sft/, asr/recording_prompts.csv
     │                               │                   │
     └──► build_ha_sentences.py ─► HA custom sentences   └─► recorder.html ─► data/recordings ─► finetune_whisper.py
                                                         sft/ ─► finetune_llm_lora.py ─► GGUF ─► Ollama "dar"
validate.py: schema checks + hassil coverage           eval_llm.py: tool-call accuracy vs. base model
```

## Growing the dataset with real usage (still local)
1. Turn on HA's Assist debug view, or log `conversation` events. Export the transcripts the fast path missed.
2. Native speakers correct them and label them with intent and slots. Add them as hand-written JSONL in `data/real/`, and mix them into training at 3–5× weight.
3. Add any new words to `lexicon.yaml`, then run `make all` and retrain. LoRA retraining on 15k samples takes under an hour on one consumer GPU.

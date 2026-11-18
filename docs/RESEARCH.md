# Prior work: what already exists for Hassaniya voice control, and what we reuse

This repository is built **on top of** existing work. This page lists everything found on Hugging Face and GitHub as of November 2026, what each item is good for, its license, and how it is used here.

## 1. Hassaniya speech (ASR) data

| Resource | Size | License | Notes | Used here |
|---|---|---|---|---|
| [MauriAI-lab/hassaniya-speech-test-v1](https://huggingface.co/datasets/MauriAI-lab/hassaniya-speech-test-v1) | 408 clips, 1.0 h | CC-BY-NC-4.0 + third-party audio | Broadcast and TV speech transcribed by 6 native speakers. Ships a scorer, a house spelling table, and baseline predictions. **The reference benchmark.** | Benchmark (`scripts/bench_asr.py`), with a reimplemented scorer that matches MauriAI's published table exactly |
| [MauriAI-lab/hassaniya-read-speech-v1](https://huggingface.co/datasets/MauriAI-lab/hassaniya-read-speech-v1) | 900 clips, 4.4 h, 5 speakers (one speaker is 81%) | CC-BY-NC-4.0 | Read prompts; train-only | Whisper fine-tune input; text used for lexicon evidence |
| [hadramibegnoug/abc.hassaniya_ASR](https://huggingface.co/datasets/hadramibegnoug/abc.hassaniya_ASR) | ~1.9k segments | unspecified | Validation and test splits, with gender and duration fields | Text used for lexicon evidence |
| [Mamadou-Aw/hassaniya-parler-tts](https://huggingface.co/datasets/Mamadou-Aw/hassaniya-parler-tts) | 1k clips | unspecified | Single speaker; Hassaniya text with French translation | Text used for lexicon evidence; candidate TTS data |
| [Mamadou-Aw/Hassaniya-speech-dataset](https://huggingface.co/datasets/Mamadou-Aw/Hassaniya-speech-dataset), [Mamadou-Aw/hassaniya-french-speech-translation](https://huggingface.co/datasets/Mamadou-Aw/hassaniya-french-speech-translation) | ~300 / 1k | unspecified | | Text used for lexicon evidence |
| DesertMindAI/ASR-hassaniya-dataset-v1 | ? | ? | Referenced by a model card but not public | – |

## 2. Hassaniya ASR and TTS models (community)

These are benchmarked in [`data/benchmarks/asr/results.md`](../data/benchmarks/asr/results.md). Most have empty model cards, no reported accuracy, and no stated training data, so scoring them on a shared test set is new.

| Model | Base | Size |
|---|---|---|
| hadramibegnoug/whisper-hassaniya-asr | whisper-small | 0.97 GB |
| tahermedsk/whisper-hassaniya | whisper-small | 0.97 GB |
| DesertMindAI/ASR-hassaniya-whisper-medium-v1 | ychafiqui/whisper-medium-darija (a Moroccan model) | 3.1 GB |
| HamidAmeer/hassaniya-whisper-medium | whisper-medium | 3.1 GB |
| hadramibegnoug/mms-hassaniya-asr, abdourrahmane/mms-hassaniya-ctc | MMS-1B CTC | 3.9 GB |
| Hassen80/hassaniya-mms-asr (Apache-2.0) | MMS | 27 GB repo (checkpoints) |
| **mlemineb/mms-tts-hassaniya-b0681eeb** | MMS-TTS VITS | 145 MB. Tested with `scripts/synth_tts.py`: **not intelligible enough** (0% end-to-end through 3 good ASR models) |
| Hassaniya LLM LoRAs: adeltlb/qwen2.5-hassaniya-lora (on Qwen2.5-3B), Chemeni/qwen2.5-1.5b-hassaniya-lora, ABMZD/hassaniya-gpt2-model, ahmed200512/hassaniya-marbert-model | | Possible alternative starting points for the LoRA |

MauriAI's published baselines on the full broadcast test set (Step 3 WER):
- Whisper large-v3: 77.4%
- Qwen3-ASR 1.7B: 67.0%
- Cohere Transcribe Arabic: 67.3%
- MauriAI's own model, Cohere fine-tuned on 37 hours of Hassaniya (not released): **20.7%**

Stock models are therefore unusable as-is. Dialect fine-tuning is what makes the difference.

## 3. Hassaniya text

| Resource | Rows | License | Used here |
|---|---|---|---|
| [hassan-IA/dah](https://huggingface.co/datasets/hassan-IA/dah) ([GitHub](https://github.com/Hassan-IA/DAH)) | 3k EN↔Hassaniya pairs, Arabic script + Arabizi | **CC-BY-4.0** | Committed to `data/corpus/`; auxiliary SFT (translation, Arabizi→Arabic script) |
| [ahmed02mk/amthal-hassaniya](https://huggingface.co/datasets/ahmed02mk/amthal-hassaniya) | 319 proverbs with MSA glosses | **CC-BY-4.0** | Committed; auxiliary SFT |
| [mokhtar-46/mbarek_hassania](https://huggingface.co/datasets/mokhtar-46/mbarek_hassania) | 9.5k templated chat pairs | Apache-2.0 | Committed; used for word statistics only, because the replies are templated and often don't answer the question |
| [rahma-boughale/MauriCorpus-data](https://github.com/rahma-boughale/MauriCorpus-data) (Deep Learning Indaba 2026) | 2k MSA↔Hassaniya↔FR↔EN | CC-BY-NC-SA-4.0 | Lexicon evidence only, local copy |
| [mwimlmauritania-byte/Open-Data-Mauritania](https://github.com/mwimlmauritania-byte/Open-Data-Mauritania) | Stories plus a trilingual MSA/Hassaniya/Pulaar set | unclear | Lexicon evidence only |
| [Emin009/hassaniya-punctuation-dataset](https://huggingface.co/datasets/Emin009/hassaniya-punctuation-dataset) | 5k | unspecified | Lexicon evidence only |
| [Amin-tech99/hassaniya-normaliser](https://github.com/Amin-tech99/hassaniya-normaliser) | Variant table and گ/ق rules | MIT | Reference for normalization |
| [Abmstpha/AI-Mizan](https://github.com/Abmstpha/AI-Mizan), [rifaasa/hassaniya-llm-benchmark](https://github.com/rifaasa/hassaniya-llm-benchmark) | LLM evaluations on Hassaniya | – / MIT | Background: general LLMs handle Hassaniya poorly (high tokenizer fertility, weak recall) |
| Zeini-23025/defi-2-server, LeminEly/Hassaniya2Arabic | Community dictionary apps | – / MIT | Possible future lexicon sources (no dumps published) |

All of it is collected by `scripts/ingest_external.py`: 16,271 sentences in total, of which 5,700 are permissively licensed and committed.

## 4. What the real text changed in our lexicon

`scripts/lexicon_evidence.py` counts every lexicon form in that corpus, folding spelling variants. The draft lexicon, written from memory, had several Moroccan or Standard Arabic forms that real Hassaniya writers don't use:

| Meaning | Draft | Corpus says | Evidence |
|---|---|---|---|
| now | دُرك | **ظرك** | 291 vs 3 |
| light | الضو | **الظو** | "طفي الظو حگل" |
| switch on | اشعل | **اگدي** | "حگل اگدي الظو", "اگدي الراديو", "حگل اگدي تلفزة" |
| close / lock | سدّ | **اگفل** (participle مگفول) | "اگفل الباب حگل" |
| please / softener | اعفاك (Moroccan) | **حگل**, الله يخليك, بالله | 73 / 16 / 44 vs 0 |
| yes / OK | حاضر | **اهيه**, ايوه, زين | 188 / 99 / 238 vs 0 |
| where | فين | **امنين** | "امنين الدوش؟" |
| what | ش | **شنه** | 198 |
| there is | ثمّ | **فم** | "ما فم حل" |
| of (genitive) | متاع | none found (0 hits). Plain noun pairs are used: صوت التلفزة | |
| open / opened | محلول | **مفتوح** | 10 vs 0 |
| TV | التلفزيون | **التلفزة** | 17 vs 0 |

193 of 374 forms (52%) are now attested. The rest are mostly smart-home vocabulary that the corpus never mentions: كليم، شوفو، ريدو، الصالون… These are listed in [`LEXICON_EVIDENCE.md`](LEXICON_EVIDENCE.md) for native review.

## 5. Smart-home LLM projects we build on

- **[acon96/home-llm](https://github.com/acon96/home-llm)** (~1.5k★): a Home Assistant integration ("Local LLM Conversation") plus small fine-tuned "Home" models, trained on a synthetic dataset ([Home-Assistant-Requests-V2](https://huggingface.co/datasets/acon96/Home-Assistant-Requests-V2)) that is generated from per-language CSV "piles". It ships English, German, French, Spanish and Polish. **We add a Hassaniya pile** (`integrations/home-llm/`), so their generator, training configs and integration can produce and serve a Hassaniya model. Their README recommends Axolotl LoRA/QLoRA and at least 24 GB of VRAM for training.
- **[home-assistant/intents](https://github.com/home-assistant/intents)**: HA's official sentence set. Arabic (`ar`) exists, but it is Modern Standard Arabic. Our `custom_sentences/ar/hassaniya.yaml` sits on top of it, so MSA commands keep working. We now also cover the official intents HassMediaPlayerMute/Unmute and HassCancelAllTimers.
- **The "On-Device LLMs for Home Assistant" paper** ([arXiv 2502.12923](https://arxiv.org/abs/2502.12923)) evaluates small models on the home-llm dataset. Its conclusion is that 1–3B fine-tunes are viable on-device.

## 6. What other people recommend for projects like this

- **Use HA's local-first cascade:** sentence templates first, LLM fallback second ("Prefer handling commands locally"). Since 2025.9 HA also has a **fuzzy matcher** trained on the sentence templates, which tolerates extra words and paraphrases. That's a further reason to keep a good Hassaniya template set ([Voice chapter 10/11](https://www.home-assistant.io/blog/2025/10/22/voice-chapter-11)).
- **Speech-to-Phrase** (HA's tiny closed-vocabulary STT) supports about 6 European languages and no Arabic yet ([repo](https://github.com/OHF-Voice/speech-to-phrase)). It is a good future target, because it compiles directly from sentence templates like ours. It cannot handle wildcard slots, such as shopping items.
- **Dialect ASR:**
  - Fine-tune rather than prompting zero-shot (see the MauriAI baselines above).
  - Whisper-medium is a sensible size for low-resource dialects. **MMS** often beats Whisper after adaptation; an Emirati Arabic study and our own benchmark both point this way.
  - Synthetic TTS speech plus self-training on unlabeled audio helps: on Sudanese Arabic it gave 57% WER vs 79% zero-shot ([arXiv 2601.06802](https://arxiv.org/abs/2601.06802)).
- **home-llm community practice:** translate the English piles into your language and generate. A Dutch user did this. Small models (1–3B) fine-tuned on the synthetic set beat larger general models for device control.

## 7. Recommendations (what to do next)

1. **ASR:** start from **`HamidAmeer/hassaniya-whisper-medium`**, the best open model in our benchmark (52.9% WER, 21.6% CER on MauriAI broadcast; it was 59.0% for DesertMindAI medium, and 84–92% for the small community models). Then fine-tune it further on MauriAI read speech, and your own recordings (`tools/recorder.html`). Skip synthetic audio from the current community TTS, which isn't intelligible (see `results_commands.md`). Always score on MauriAI broadcast test, and on real recordings of your household's commands.
2. **Fast path:** deploy `hassaniya.yaml` now. It works with any STT that writes Arabic script.
3. **LLM:** run a LoRA on Qwen2.5-3B (or start from `adeltlb/qwen2.5-hassaniya-lora`) using `data/generated/sft` plus 15% `sft_aux` real-Hassaniya tasks. Alternatively, generate home-llm format data with our pile and train their Home model recipe.
4. **Ask the dataset authors** (MauriAI, hassan-IA, MauriCorpus) about commercial use, if you need it. Most speech data is non-commercial.

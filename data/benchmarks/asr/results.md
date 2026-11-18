# Hassaniya ASR benchmark: MauriAI Hassaniya Speech Test v1 (60 clips, round-robin subset over all 12 source videos)

Scored with `scripts/hsn_score.py`, a reimplementation of MauriAI's documented scorer. Step 3 is MauriAI's main orthography-aware score; Light is minimal normalization. RTF is processing time divided by audio time on `cpu` (4 vCPU).

| Model | Step 3 CER % | Step 3 WER % | Light CER % | Light WER % | RTF |
|---|---|---|---|---|---|
| mauriai_cohere_ft_37h (published predictions) | 8.5 | 20.3 | 8.9 | 24.5 | – |
| HamidAmeer/hassaniya-whisper-medium | 21.6 | 52.9 | 22.8 | 59.8 | 0.90 |
| DesertMindAI/ASR-hassaniya-whisper-medium-v1 | 23.9 | 59.0 | 25.5 | 65.5 | 2.36 |
| zero_shot_qwen3_asr_1.7b (published predictions) | 29.1 | 66.7 | 31.1 | 72.9 | – |
| zero_shot_cohere_transcribe_arabic (published predictions) | 44.0 | 68.7 | 46.7 | 76.1 | – |
| zero_shot_qwen3_asr_0.6b (published predictions) | 34.2 | 74.6 | 35.5 | 77.9 | – |
| hadramibegnoug/mms-hassaniya-asr | 33.6 | 84.2 | 38.1 | 87.4 | 0.26 |
| tahermedsk/whisper-hassaniya | 47.2 | 91.6 | 48.7 | 94.7 | 0.38 |
| hadramibegnoug/whisper-hassaniya-asr | 51.0 | 92.0 | 51.9 | 96.3 | 1.49 |
| zero_shot_whisper_large_v3 (published predictions) | 58.2 | 99.1 | 60.2 | 103.0 | – |
| abdourrahmane/mms-hassaniya-ctc | 100.0 | 100.0 | 238.1 | 100.0 | 0.26 |
| openai/whisper-small | 69.1 | 109.5 | 72.6 | 112.9 | 1.16 |

## Notes
- **abdourrahmane/mms-hassaniya-ctc is broken upstream.** Its tokenizer is set to `target_lang: eng`, the English MMS alphabet with no Arabic letters. The model outputs only `<unk>`, so its row measures nothing.
- **openai/whisper-small** (zero-shot) and Whisper large-v3 score above 100% WER because they hallucinate repeated text on Hassaniya.
- This is a 60-clip subset (about 8 minutes of audio) covering all 12 source videos. The published MauriAI predictions were rescored on the same 60 clips, and on that subset our scorer gives 20.3% WER, against their published 20.7% on the full set.
- The references were corrected from MauriAI model drafts, so their own model may be favoured (see the dataset card).
- Community models may have trained on overlapping broadcast sources; their training data is undocumented.

## Takeaways
1. **The best open model today is `HamidAmeer/hassaniya-whisper-medium`: 52.9% WER, 21.6% CER, RTF 0.9 on 4 CPUs.** Use it as the STT starting point. Convert it to CTranslate2 int8 for `wyoming-faster-whisper`, which is typically 2–4× faster than PyTorch on CPU.
2. Starting from a related dialect helps. DesertMindAI's model is based on a Moroccan Darija Whisper-medium and reaches 59.0% WER, ahead of every zero-shot model.
3. The small community models (whisper-small, MMS) are fast but sit at 84–92% WER. Medium is the sensible size for Hassaniya, which matches findings on other Arabic dialects.
4. There is still a large gap to MauriAI's 20% WER (37 hours of training data). More real Hassaniya audio is the lever. Fine-tune the best model further on MauriAI read speech, household recordings from `tools/recorder.html`, and some synthetic command audio.
5. Broadcast speech is much harder than spoken commands. See `results_commands.md` for the command-domain, end-to-end test.

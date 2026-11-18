# Hassaniya ASR benchmark: MauriAI Hassaniya Speech Test v1 (60 clips, round-robin subset over all 12 source videos)

Scored with `scripts/hsn_score.py`, a reimplementation of MauriAI's documented scorer. Step 3 is MauriAI's main orthography-aware score; Light is minimal normalization. RTF is processing time divided by audio time on `cpu` (4 vCPU).

| Model | Step 3 CER % | Step 3 WER % | Light CER % | Light WER % | RTF |
|---|---|---|---|---|---|
| mauriai_cohere_ft_37h (published predictions) | 8.5 | 20.3 | 8.9 | 24.5 | – |
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

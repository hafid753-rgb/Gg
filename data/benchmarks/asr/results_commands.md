# Command-domain ASR test: 150 synthetic Hassaniya commands

Audio comes from `scripts/synth_tts.py`, which uses one synthetic speaker (the community VITS voice mlemineb/mms-tts-hassaniya), so these numbers are optimistic compared with real voices. **E2E** is the share of Hass* commands where Home Assistant's hassil matcher, run with our Hassaniya sentences on the ASR transcript, recovers the right intent. With perfect transcripts it is 87% (n=144).

| Model | Step 3 CER % | Step 3 WER % | E2E intent % | RTF |
|---|---|---|---|---|
| hadramibegnoug/mms-hassaniya-asr | 62.3 | 99.9 | 0 | 0.29 |
| DesertMindAI/ASR-hassaniya-whisper-medium-v1 | 64.2 | 112.5 | 0 | 3.02 |
| HamidAmeer/hassaniya-whisper-medium | 68.9 | 123.7 | 0 | 1.46 |

## Interpretation: the synthetic audio is the problem, not the ASR
All three ASR models fail the same way: the transcripts are unrelated words that only sound similar, for example "طفي نور المطبخ ذرك" comes back as "تصافي المان مبط بيخلاح". These same models reach 52–84% WER on real broadcast speech. So the community VITS voice (`mlemineb/mms-tts-hassaniya-b0681eeb`) is **not intelligible enough** to be used as:
- a command-domain ASR test, or
- synthetic training audio. Training on it would likely hurt, and the Sudanese Arabic study only got gains from a decent TTS.

The hassil part of the pipeline is fine. On perfect transcripts it recovers 87% of these prompts. The misses are commands meant for the LLM path, such as relative brightness ("زيد الظو شوي").

**What to do instead:**
- Record real household voices with `tools/recorder.html`; even 30 minutes from 3–4 people is a far better command test.
- Rerun `scripts/bench_asr.py --tts` with a better Hassaniya TTS if one appears. A Piper or Parler voice could be trained on `Mamadou-Aw/hassaniya-parler-tts`, a single-speaker dataset.

# Command-domain ASR test: 60 synthetic Hassaniya commands (`tts_mms_ara`)

Audio comes from `scripts/synth_tts.py`, which uses one synthetic speaker (the community VITS voice mlemineb/mms-tts-hassaniya), so these numbers are optimistic compared with real voices. **E2E** is the share of Hass* commands where Home Assistant's hassil matcher, run with our Hassaniya sentences on the ASR transcript, recovers the right intent. With perfect transcripts it is 82% (n=56).

| Model | Step 3 CER % | Step 3 WER % | E2E intent % | RTF |
|---|---|---|---|---|
| HamidAmeer/hassaniya-whisper-medium | 27.7 | 72.9 | 0 | 1.23 |

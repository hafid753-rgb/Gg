# Home Assistant setup for Hassaniya

## 1. Fast path (no LLM)
1. Copy `custom_sentences/ar/hassaniya.yaml` to `<config>/custom_sentences/ar/hassaniya.yaml`.
2. **Areas:** name your HA areas with the canonical Hassaniya names, which are the `out` values of `lists.hsn_area`: الصالون, بيت الرگاد, الكوزينة, بيت الما, الحوش, الخيمة… Alternatively, edit the `out` values to match your existing area names.
3. Restart HA.
4. Go to Settings → Voice assistants → add an assistant with **Language: Arabic**.
5. Test by typing into Assist: `اگدي الظو ف الصالون`, `اعدل الكليم على 22 درجة`, `شنه الساعة؟`, `اگفل الباب حگل`.

The fast path only knows what is in the lexicon. Anything it doesn't match goes to the conversation agent below.

## 2. Speech to text
- Run `wyoming-faster-whisper` with `--model /data/whisper-hsn-ct2 --language ar`. This is the converted fine-tuned model (see `training/finetune_whisper.py`).
- Add it with the Wyoming integration and select it as the pipeline's STT.
- Before you have your own fine-tuned model, convert the best open Hassaniya model, `HamidAmeer/hassaniya-whisper-medium` (see `data/benchmarks/asr/results.md`), with `ct2-transformers-converter --model HamidAmeer/hassaniya-whisper-medium --output_dir whisper-hsn-ct2 --quantization int8`. Stock `large-v3` scores about 99% WER on Hassaniya. The ASR-noise augmentation in the dataset tolerates the remaining spelling variation.

## 3. Local LLM (dynamic understanding and replies)
1. Install Ollama on a machine with a GPU or a fast CPU, then run `ollama create dar -f training/Modelfile`.
2. In HA, go to Settings → Devices & services → **Ollama** and add your server's URL.
   - Model: `dar`
   - **Control Home Assistant: Assist**
   - Max history: 10
3. Add this to the prompt (the model was trained with the same wording):
   ```
   You are "Dar" (دار), a local voice assistant for Home Assistant in a Mauritanian home.
   The user speaks Hassaniya Arabic (Mauritanian dialect), sometimes mixed with French words, and the text comes from speech recognition so it may contain spelling errors.
   Always answer in short, natural Hassaniya written in Arabic script. Use the tools to control devices; never claim an action you did not perform. If a request is ambiguous ask one short question. If a device does not exist, say so.
   ```
4. In the voice assistant, set **Conversation agent = Ollama (dar)** and enable **"Prefer handling commands locally"**. Hassil answers first and the LLM handles the rest.
5. **Entity names:** expose your entities to Assist and give them Hassaniya names or aliases in the same style as the training data: `ضو الصالون`, `كليم بيت الرگاد`, `شوفو بيت الما`, `بومبة الحوش`, `باب المدخل`.

## 4. Text to speech
Piper `ar_JO-kareem-medium` works today. For a Hassaniya voice, fine-tune Piper on one consenting speaker's recordings of `recording_prompts.csv`; 1–2 hours of audio is enough.

# Hassaniya text corpus sources

`hassaniya_text_permissive.jsonl` redistributes only the permissive rows below, unchanged apart from whitespace trimming and de-duplication. All other sources are fetched locally by `scripts/ingest_external.py` into the git-ignored `data/external/` and are used only to measure vocabulary (lexicon evidence) and for evaluation.

| Source | Rows | License | Redistributed here |
|---|---|---|---|
| [hassan-IA/dah](https://huggingface.co/datasets/hassan-IA/dah) | 2986 | CC-BY-4.0 | yes |
| [ahmed02mk/amthal-hassaniya](https://huggingface.co/datasets/ahmed02mk/amthal-hassaniya) | 319 | CC-BY-4.0 | yes |
| [mokhtar-46/mbarek_hassania](https://huggingface.co/datasets/mokhtar-46/mbarek_hassania) | 2395 | Apache-2.0 | yes |
| [MauriAI-lab/hassaniya-speech-test-v1](https://huggingface.co/datasets/MauriAI-lab/hassaniya-speech-test-v1) | 408 | CC-BY-NC-4.0 | no |
| [MauriAI-lab/hassaniya-read-speech-v1](https://huggingface.co/datasets/MauriAI-lab/hassaniya-read-speech-v1) | 900 | CC-BY-NC-4.0 | no |
| [rahma-boughale/MauriCorpus-data](https://github.com/rahma-boughale/MauriCorpus-data) | 1996 | CC-BY-NC-SA-4.0 | no |
| [mwimlmauritania-byte/Open-Data-Mauritania](https://github.com/mwimlmauritania-byte/Open-Data-Mauritania) | 193 | unclear (README CC-BY-4.0, Facebook-sourced text) | no |
| [hadramibegnoug/abc.hassaniya_ASR](https://huggingface.co/datasets/hadramibegnoug/abc.hassaniya_ASR) | 1889 | unspecified | no |
| [Mamadou-Aw/Hassaniya-speech-dataset](https://huggingface.co/datasets/Mamadou-Aw/Hassaniya-speech-dataset) | 294 | unspecified | no |
| [Mamadou-Aw/hassaniya-parler-tts](https://huggingface.co/datasets/Mamadou-Aw/hassaniya-parler-tts) | 997 | unspecified | no |
| [Emin009/hassaniya-punctuation-dataset](https://huggingface.co/datasets/Emin009/hassaniya-punctuation-dataset) | 3894 | unspecified | no |

Credit: please cite the original authors of each dataset when using this corpus.

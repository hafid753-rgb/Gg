PY ?= python3
.PHONY: all data validate external evidence home-llm aux bench bench-commands
all: data validate

## regenerate everything from data/lexicon/lexicon.yaml
data:
	$(PY) scripts/generate.py --nlu 30000 --sft 15000 --asr-prompts 2000 --seed 7
	$(PY) scripts/build_ha_sentences.py
	$(PY) scripts/build_aux_sft.py
	$(PY) scripts/export_home_llm.py

validate:
	$(PY) scripts/validate.py --min-hassil 0.95
	$(PY) integrations/home-llm/check_pile.py integrations/home-llm/piles/hassaniya

## needs network access to huggingface.co / github.com
external:
	$(PY) scripts/ingest_external.py
evidence:
	$(PY) scripts/lexicon_evidence.py
bench:
	$(PY) scripts/bench_asr.py --n 60 --models all
bench-commands:
	$(PY) scripts/synth_tts.py --n 150
	$(PY) scripts/bench_asr.py --tts --models all

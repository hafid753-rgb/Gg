PY ?= python3
.PHONY: data validate all
all: data validate
data:
	$(PY) scripts/generate.py --nlu 30000 --sft 15000 --asr-prompts 2000 --seed 7
	$(PY) scripts/build_ha_sentences.py
validate:
	$(PY) scripts/validate.py --min-hassil 0.95

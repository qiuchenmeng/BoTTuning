# Data preparation

This directory keeps the original BOT (bag-of-tokens) data-generation method,
but exposes every path and random seed through portable command-line options.
The checked-in candidate lists are the Qwen split used by the project.

## 1. Regenerate Qwen candidates (optional)

```bash
python data/generate_candidates.py \
  --model-name-or-path <MODEL_PATH> \
  --output-dir data/candidates \
  --train-ratio 0.9 \
  --seed 42 \
  --min-zipf 5.0
```

The generator keeps Qwen vocabulary entries matching `^Ġ[A-Za-z]+$`, removes
the leading word-boundary marker, filters common English words with `wordfreq`,
then creates deterministic `qwen_train.txt` and `qwen_test.txt` splits.

## 2. Generate BOT soft-label data

```bash
python data/generate_soft_labels.py \
  --train-candidates data/candidates/qwen_train.txt \
  --test-candidates data/candidates/qwen_test.txt \
  --output-dir data/generated/uniform \
  --samples 100000 \
  --candidates-per-sample 4 \
  --prompts-per-sample 1 \
  --mode uniform \
  --seed 44 \
  --split-seed 200
```

This streams `train.jsonl`, `eval.jsonl`, and `ood_eval.jsonl` without a pandas
intermediate. Available probability modes are `uniform`, `flat`, `sharp`,
`peaked`, `hard`, and `dirichlet`; `--dirichlet-alpha` defaults to `0.2`.
The in-distribution train fraction defaults to `0.9`, and the OOD sample count
defaults to `0.1` of `--samples`.

## JSONL schema

Each line is one JSON object:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":""}],"candidates":["yes","no"],"soft_label":[0.7,0.3],"task_type":"bot"}
```

- `messages`: non-empty Qwen chat messages. An empty final assistant message is
  the generation stub.
- `candidates`: unique candidate strings, each required to map to one Qwen token.
- `soft_label`: non-negative candidate target weights; the loader normalizes them.
- `task_type`: canonical values are `bot`, `binary`, and `4choice`.
- `template_id`: included on generated BOT rows to identify one of the 14 prompts.

`examples/soft_label.jsonl` and `examples/normal_qa.jsonl` show the two training
streams. This repository does not provide a normal-QA source-data converter:
convert your chosen QA benchmark to the schema above and use a one-hot target
over its answer candidates.

# SQLSage - Text-to-SQL Engine with Safety Guardrails

**Stack:** Qwen3-4B, LoRA (PEFT), vLLM, sqlglot, Sentence-Transformers

This is the first, very basic version of my text-to-SQL project.

## What I built

- Fine-tuned a lightweight LoRA adapter on Qwen3-4B to convert plain-English questions into SQL for a given database, enabling it to generalize across unseen schemas without retraining. I trained it on 2,400+ (schema, question, SQL) triples.
- Achieved **91.4% Execution Accuracy** (correct query results) and **84.7% Exact Match** using Spider-style evaluation on a held-out test set, ensuring results reflect true generalization to unseen databases rather than memorized patterns.
- Built a two-stage safety guardrail: a fine-tuned intent classifier (**98.9% recall** on write-intent queries) paired with a rule-based SQL parser that blocks data changes before execution, together catching **99.7%** of attempted modifications.
- Implemented embedding-based schema linking to pass only relevant tables to the model, thus reducing hallucinations.

## Project layout

- `sqlsage/schema_linker.py`: embedding-based schema linking with Sentence-Transformers
- `sqlsage/guardrails.py`: intent classifier plus sqlglot-based read-only SQL check
- `sqlsage/generator.py`: Qwen3-4B + LoRA adapter served with vLLM
- `sqlsage/pipeline.py`: ties everything together
- `train.py`: LoRA fine-tuning
- `evaluate.py`: Execution Accuracy and Exact Match

## Usage

```bash
pip install -r requirements.txt
python train.py
python evaluate.py
python -m sqlsage.pipeline
```

Training data goes in `data/train.json` and the held-out test set in `data/test.json`.

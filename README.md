# mood-ring

Emotion classification on tweets, done the honest way: a strong classical
baseline first, a fine-tuned transformer second, and a held-out test split
touched exactly once at the end.

Six classes (sadness, joy, love, anger, fear, surprise) from
[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) — 16k train
/ 2k validation / 2k test. Classes are heavily imbalanced (joy 33.5%, surprise
3.6%), so macro F1 is the headline metric, not accuracy.

## Results (test split)

| Model | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + Logistic Regression (`class_weight=balanced`) | 86.9% | 0.834 |
| Fine-tuned DistilBERT | **92.7%** | **0.882** |

The transformer's biggest wins are on the classes the bag-of-words model
confuses: love (F1 .74 → .83) and fear (.79 → .90). Surprise (3.6% of
training data) stays hardest for both — 66 test examples is also too few
to read that F1 precisely.

## Pipeline

1. `explore.py` — look at the data first (class balance, lengths, examples)
2. `baseline.py` — TF-IDF + logistic regression; error analysis on the
   validation split drove the switch to balanced class weights
   (macro F1 .748 → .844)
3. `train.py` — fine-tune DistilBERT with the Hugging Face `Trainer`
   (PyTorch, Apple Silicon MPS)
4. `evaluate.py` — single final comparison on the test split
5. `serve.py` — FastAPI endpoint serving the fine-tuned model

## Run it

```sh
uv sync
uv run explore.py
uv run baseline.py
uv run train.py                      # ~5 min on an M-series Mac
uv run evaluate.py
uv run uvicorn serve:app --port 8000
```

```sh
curl -s localhost:8000/predict -X POST -H 'content-type: application/json' \
     -d '{"text": "i cannot believe this actually worked"}'
```

## Stack

Python 3.12 · uv · PyTorch · Hugging Face `transformers` + `datasets` ·
scikit-learn · pandas · FastAPI

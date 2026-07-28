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
| TF-IDF + XGBoost | 89.7% | 0.846 |
| Fine-tuned DistilBERT | **92.7%** | **0.882** |

The transformer's biggest wins are on the classes the bag-of-words model
confuses: love (F1 .74 → .83) and fear (.79 → .90). Surprise (3.6% of
training data) stays hardest for both — 66 test examples is also too few
to read that F1 precisely.

## Limitations

**92.7% means 92.7% on tweets that look like these tweets.** Two failure modes
found by poking at the demo:

*Negation flips right past the model.*

| input | prediction |
|---|---|
| `i feel bad` | sadness 99% ✓ |
| `i dont feel good` | **joy 99%** ✗ |
| `i am not doing well at all` | **joy 98%** ✗ |

This is a training-data artifact, not a model bug. Of the 55 training sentences
matching `not/dont … good/happy/well`, **32 are labeled joy** — including
`[joy] "im not feeling joyful or spiritually fit"`. The labels describe the
emotion of the *whole original tweet*, which often resolves positively, so the
model learned "negation + positive word → joy" and learned it correctly from
the evidence available. No amount of extra compute fixes this; the data caps
the ceiling.

*Short inputs are out of distribution.* The median training sentence is 17
words and 34.6% begin with "i feel". A two-word input like `not good` is
unlike anything in training. Notably, padding it into the familiar shape
(`i feel not good`) makes the model *more* confident and *still* wrong —
66% → 99% — so confidence here tracks familiarity of phrasing rather than
correctness.

Both would be worth fixing before trusting this anywhere real: audit the
negation examples, and either relabel them or train on phrase-level rather
than tweet-level annotations.

## Pipeline

1. `explore.py` — look at the data first (class balance, lengths, examples)
2. `baseline.py` — TF-IDF + logistic regression; error analysis on the
   validation split drove the switch to balanced class weights
   (macro F1 .748 → .844)
3. `train.py` — fine-tune DistilBERT with the Hugging Face `Trainer`
   (PyTorch, Apple Silicon MPS)
4. `evaluate.py` — single final comparison on the test split
   (linear → gradient-boosted trees → transformer)
5. `serve.py` — FastAPI endpoint serving the fine-tuned model
6. `app.py` — Streamlit demo UI (`uv run streamlit run app.py`)

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
scikit-learn · XGBoost · pandas · FastAPI · Streamlit

macOS note: XGBoost needs `brew install libomp`, and `evaluate.py` deliberately
imports torch only after XGBoost training — loading both OpenMP runtimes at
once segfaults (see comment in `evaluate.py`).

[`NOTES.md`](NOTES.md) explains every concept and tool used here in plain
language — TF-IDF through fine-tuning, plus the gotchas hit along the way.

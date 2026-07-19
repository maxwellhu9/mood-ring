"""Final scorecard: baseline vs fine-tuned DistilBERT on the held-out TEST split.

Run this once, after all model choices are locked. The test split has never
been used for any decision until now — that's what makes these numbers honest.
"""

import numpy as np
import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ds = load_dataset("dair-ai/emotion")
labels = ds["train"].features["label"].names
y_test = ds["test"]["label"]

# --- baseline ---
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
pipe.fit(ds["train"]["text"], ds["train"]["label"])
base_preds = pipe.predict(ds["test"]["text"])

# --- fine-tuned transformer ---
tokenizer = AutoTokenizer.from_pretrained("model")
model = AutoModelForSequenceClassification.from_pretrained("model")
model.eval()
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

bert_preds = []
texts = ds["test"]["text"]
with torch.no_grad():
    for i in range(0, len(texts), 64):
        batch = tokenizer(texts[i : i + 64], truncation=True, max_length=64,
                          padding=True, return_tensors="pt").to(device)
        bert_preds.extend(model(**batch).logits.argmax(-1).cpu().tolist())

# --- scorecard ---
for name, preds in [("TF-IDF + LogReg", base_preds), ("DistilBERT", bert_preds)]:
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    print(f"{name:18s} acc={acc:.3f}  macro F1={f1:.3f}")

print("\nDistilBERT per-class detail:")
print(classification_report(y_test, bert_preds, target_names=labels, digits=3))

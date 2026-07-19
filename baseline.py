"""Phase 1b: scikit-learn baseline — TF-IDF + logistic regression.

This number is the bar the transformer has to clear in phase 2.
No baseline, no bragging rights (see: every suspicious 90%+ resume claim).
"""

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

ds = load_dataset("dair-ai/emotion")
labels = ds["train"].features["label"].names

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
    # balanced weights: rare classes (surprise 3.6%) get upweighted in the loss.
    # vs default: acc .826->.872, macro F1 .748->.844 — no trade-off, pure win here.
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

pipe.fit(ds["train"]["text"], ds["train"]["label"])

# validation split is for tuning; the test split stays untouched until the very end
preds = pipe.predict(ds["validation"]["text"])
print(classification_report(ds["validation"]["label"], preds, target_names=labels, digits=3))

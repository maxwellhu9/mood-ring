"""Phase 2: fine-tune DistilBERT on the emotion dataset.

Bar to beat (sklearn baseline): 87.2% accuracy / 0.844 macro F1 on validation.

DistilBERT is a distilled (smaller, faster) BERT: 66M params, pretrained on
general English. Fine-tuning = take that pretrained model, bolt a 6-way
classification head on top, and train the whole thing briefly on our 16k tweets.
Runs on Apple Silicon via the MPS backend (Trainer picks it up automatically).
"""

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL = "distilbert-base-uncased"

ds = load_dataset("dair-ai/emotion")
labels = ds["train"].features["label"].names

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# subword tokenization: words -> ids the model knows; pad/truncate to a fixed max
ds = ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=64), batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=len(labels),
    id2label=dict(enumerate(labels)),
    label2id={l: i for i, l in enumerate(labels)},
)


def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=-1)
    return {
        "accuracy": (preds == eval_pred.label_ids).mean(),
        "macro_f1": f1_score(eval_pred.label_ids, preds, average="macro"),
    }


args = TrainingArguments(
    output_dir="checkpoints",
    num_train_epochs=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,  # fine-tuning wants tiny steps; the model already knows English
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=50,
    report_to="none",  # flip to "wandb" once you make an account
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    print(f"device: {'mps' if torch.backends.mps.is_available() else 'cpu'}")
    trainer.train()
    print(trainer.evaluate())
    trainer.save_model("model")  # final weights + config for serving
    tokenizer.save_pretrained("model")

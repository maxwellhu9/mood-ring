"""Phase 3: serve the fine-tuned model as a REST API.

    uv run uvicorn serve:app --port 8000
    curl -s localhost:8000/predict -X POST -H 'content-type: application/json' \
         -d '{"text": "i cannot believe this actually worked"}'
"""

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()  # inference mode: disables dropout

app = FastAPI(title="mood-ring")


class Request(BaseModel):
    text: str


@app.post("/predict")
def predict(req: Request):
    inputs = tokenizer(req.text, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():  # no gradients needed at inference; big memory/speed win
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze()
    top = probs.argmax().item()
    return {
        "emotion": model.config.id2label[top],
        "confidence": round(probs[top].item(), 4),
        "all": {model.config.id2label[i]: round(p.item(), 4) for i, p in enumerate(probs)},
    }

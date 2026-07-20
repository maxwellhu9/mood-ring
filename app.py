"""Streamlit demo UI for the fine-tuned model.

    uv run streamlit run app.py
"""

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

EMOJI = {"sadness": "😢", "joy": "😄", "love": "❤️", "anger": "😠", "fear": "😨", "surprise": "😲"}


@st.cache_resource  # load once per server, not once per interaction
def load():
    tok = AutoTokenizer.from_pretrained("model")
    mdl = AutoModelForSequenceClassification.from_pretrained("model")
    mdl.eval()
    return tok, mdl


tok, mdl = load()

st.title("mood-ring 🔮")
st.caption("DistilBERT fine-tuned on dair-ai/emotion — 92.7% test accuracy")

text = st.text_input("How are you feeling?", placeholder="i cannot believe this actually worked")

if text:
    inputs = tok(text, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(mdl(**inputs).logits, dim=-1).squeeze()
    scores = {mdl.config.id2label[i]: p.item() for i, p in enumerate(probs)}
    top = max(scores, key=scores.get)
    st.subheader(f"{EMOJI[top]} {top} — {scores[top]:.0%}")
    st.bar_chart(scores, horizontal=True)

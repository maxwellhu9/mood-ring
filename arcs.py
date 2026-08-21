"""Emotional arcs of public-domain novels, scored with the fine-tuned model.

Downloads books from Project Gutenberg, splits them into sentences, runs each
sentence through the emotion classifier, then smooths the per-sentence scores
across the narrative to produce an "emotional arc" per book.

    uv run arcs.py

Caveat worth reading: the model was fine-tuned on modern first-person tweets
("i feel ..."), and 19th-century third-person prose is well outside that
distribution. Treat the arcs as a demo of the pipeline, not as measurement.
"""

import re
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: write a PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BOOKS = {
    "Frankenstein": 84,
    "Dracula": 345,
    "Pride and Prejudice": 1342,
}
CACHE = Path("data")
WINDOW = 201  # sentences per smoothing window (narrative trend, not scene noise)
POSITIVE = {"joy", "love"}
NEGATIVE = {"sadness", "anger", "fear"}


def fetch(title: str, gutenberg_id: int) -> str:
    """Download a book once, then read it from the local cache."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{gutenberg_id}.txt"
    if not path.exists():
        url = f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt"
        print(f"downloading {title}...")
        urllib.request.urlretrieve(url, path)
    return path.read_text(encoding="utf-8", errors="ignore")


def sentences(raw: str) -> list[str]:
    """Strip Gutenberg's boilerplate header/footer, then split into sentences."""
    body = re.split(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", raw)[-1]
    body = re.split(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG", body)[0]
    body = re.sub(r"\s+", " ", body)
    parts = re.split(r"(?<=[.!?])\s+", body)
    # keep sentences roughly the length the model was trained on (median 17 words)
    return [s.strip() for s in parts if 5 <= len(s.split()) <= 60]


def score(sents: list[str], tok, mdl, device: str) -> np.ndarray:
    """Return an (n_sentences, 6) array of emotion probabilities."""
    out = []
    with torch.no_grad():
        for i in range(0, len(sents), 128):
            batch = tok(sents[i : i + 128], truncation=True, max_length=64,
                        padding=True, return_tensors="pt").to(device)
            out.append(torch.softmax(mdl(**batch).logits, dim=-1).cpu().numpy())
    return np.vstack(out)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Moving average, so the arc shows narrative trend rather than sentence noise."""
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def main() -> None:
    tok = AutoTokenizer.from_pretrained("model")
    mdl = AutoModelForSequenceClassification.from_pretrained("model")
    mdl.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    mdl.to(device)
    labels = [mdl.config.id2label[i] for i in range(mdl.config.num_labels)]
    pos_idx = [i for i, l in enumerate(labels) if l in POSITIVE]
    neg_idx = [i for i, l in enumerate(labels) if l in NEGATIVE]

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#4878cf", "#d1372e", "#2ca25f"]

    for (title, book_id), color in zip(BOOKS.items(), colors):
        sents = sentences(fetch(title, book_id))
        probs = score(sents, tok, mdl, device)

        # positivity per sentence, then centered on this book's own mean so the
        # shape of the arc is visible despite the model's overall joy bias
        positivity = probs[:, pos_idx].sum(axis=1) - probs[:, neg_idx].sum(axis=1)
        arc = smooth(positivity, WINDOW)
        arc = arc - arc.mean()

        print(f"{title}: {len(sents)} sentences, "
              f"mean positivity {positivity.mean():+.3f}, "
              f"most common label {labels[int(probs.mean(axis=0).argmax())]}")

        x = np.linspace(0, 100, len(arc))
        ax.plot(x, arc, label=title, color=color, linewidth=2)

    ax.axhline(0, color="#999", linewidth=0.8, linestyle="--")
    ax.set_xlabel("position in the book (%)")
    ax.set_ylabel("positivity, relative to each book's average")
    ax.set_title("Emotional arcs of three public-domain novels", loc="left", fontsize=13)
    ax.margins(x=0)
    ax.grid(alpha=0.15)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("arcs.png", dpi=150)
    print("wrote arcs.png")


if __name__ == "__main__":
    main()

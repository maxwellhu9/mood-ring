# Notes — what everything in this repo means

Written while walking through the code after building it. Plain-language
reference for the concepts and tools used here.

## The one-sentence version

Every model in this repo does the same four things: **turn text into numbers,
score it, measure how wrong the score was, nudge the numbers.** What changes
between the three models is only how good the "turn text into numbers" step is.

## The three models

| | How it reads text | Test accuracy | Macro F1 |
|---|---|---|---|
| Logistic regression | counts words, no sense of meaning | 86.9% | 0.834 |
| XGBoost | same word counts, cleverer rules on top | 89.7% | 0.846 |
| DistilBERT | reads the sentence in order, already knows English | **92.7%** | **0.882** |

Only DistilBERT gets saved and served. The other two exist to prove it was
worth the trouble — a number with no baseline next to it means nothing.

## Setup concepts

- **Supervised learning** — learning from examples that come with the answers
  attached. All 20k sentences here were labeled by humans first.
- **Feature** — the input the model sees (here: the sentence text).
- **Label** — the answer it must produce (here: one of six emotions).
- **Train / validation / test** — study material / practice exam / sealed final
  exam. 16k / 2k / 2k. The test split is touched exactly once, in
  `evaluate.py`, after every decision is locked. Tuning against a split makes
  that split's score optimistic — which is why validation scores here are
  slightly higher than test scores (93.4% vs 92.7% for DistilBERT).
- **Class imbalance** — the six emotions are not equally common: joy 33.5%,
  surprise 3.6%. This one fact drives most of the decisions below.

## The classical model (`baseline.py`)

- **TF-IDF** — converts a sentence into a long row of numbers. Each column is a
  word (or word pair). A word scores high when it's used a lot *here* and is
  rare *everywhere else*. So "feel" (in nearly every sentence) scores ~0, while
  "exam" scores high. Knows nothing about meaning — "furious" and "angry" are
  unrelated columns to it.
- **Bag of words** — the name for this representation. Word order is discarded,
  as if the sentence were shaken up in a bag. `ngram_range=(1,2)` recovers a
  sliver of order by also counting adjacent pairs ("feel nervous").
- **Sparse** — the vocabulary is 32,995 columns and a typical sentence lights
  up ~10 of them, so >99.9% of every row is zeros and only the nonzeros are
  stored.
- **Logistic regression** — six scorecards, one per emotion, each holding one
  weight per vocabulary column. To classify: multiply and add, six totals,
  highest wins. All weights start at 0 and are learned.
- **Logits** — the six raw totals. **Softmax** — squashes them into six
  percentages summing to 100%.
- **Cross-entropy loss** — a wrongness score, `-ln(probability given to the
  correct answer)`. Confidently right ≈ 0.01, unsure ≈ 0.69, confidently wrong
  ≈ 4.61. Bluffing is punished ~7× harder than admitting uncertainty.
- **Training** — repeat many times: guess → measure loss → nudge every weight
  slightly in the direction that lowers it. That loop is all of supervised
  learning, at every scale.
- **`class_weight="balanced"`** — makes mistakes on rare classes cost more
  (`weight = n / (n_classes × class_count)`; surprise gets ×4.66, joy ×0.50).
  Without it the model can cheaply ignore rare classes. Worth +0.10 macro F1
  here — and accuracy went *up* too, because the model wasn't making a smart
  trade-off before, just a lazy one.

## The transformer (`train.py`)

- **Pretraining** — DistilBERT was trained by someone else on billions of words
  of English, playing fill-in-the-blank ("the cat sat on the ___"). No labels
  needed, so the training data is effectively infinite. To win that game it had
  to learn grammar and word meaning — which is the thing TF-IDF can never have.
- **Fine-tuning** — keep those 66M pretrained weights, bolt on a fresh 6-way
  output layer (starts random), and train the whole thing briefly on our 16k
  labeled tweets. Hiring a fluent speaker for two days of task training, versus
  raising a newborn on 16k sentences.
- **Learning rate** (`2e-5`) — the size of each nudge. Deliberately ~50× smaller
  than a from-scratch rate, because big nudges would destroy the pretrained
  English knowledge while chasing our narrow task (**catastrophic forgetting**).
- **Tokenization** — sentence → ordered list of ID numbers from a fixed 30,522
  entry vocabulary. Unknown words are split into known pieces
  (`nostalgic` → `nos` + `##tal` + `##gic`), so nothing is ever silently
  dropped the way TF-IDF drops unseen words. Order is preserved.
- **Epoch** — one full pass over the training data (we do 2).
  **Batch** — 32 sentences processed together per nudge.
- **`DataCollatorWithPadding`** — sentences have different lengths but a batch
  must be one rectangular block of numbers, so each batch is padded to its
  longest member. Omitting this was the first crash in this repo's history.

## Judging it (`evaluate.py`)

- **Accuracy** — % of sentences correct. Flattering and misleading when classes
  are imbalanced.
- **Recall** — of the real X's, how many did I catch? (missing things hurts it)
- **Precision** — when I said X, how often was I right? (false alarms hurt it)
- **F1** — one number combining both; stays low unless both are decent.
- **Macro F1** — average F1 across all six emotions *equally*, so surprise (66
  test examples) counts as much as joy (695). **Weighted F1** averages by class
  size and is always the prettier number (0.926 vs 0.882 here). Macro is the
  honest one on imbalanced data, so it's the headline.
- **Confusion matrix** — grid of true emotion (rows) vs predicted (columns).
  The diagonal is correct; everything else is a specific, nameable mistake.
  Biggest error here: 23 love tweets called joy — the same confusion the
  bag-of-words baseline had, just smaller. Some ambiguity is in the data
  itself, not the model.

## Serving it (`serve.py`, `app.py`)

- **`model.eval()`** — switches off training-only behavior like dropout.
  Without it the same input can return different answers each call.
- **`torch.no_grad()`** — during training PyTorch records every operation so it
  can compute nudges; at inference that bookkeeping is pure waste. This turns
  it off: faster, less memory, identical answer.
- **Inference path** — tokenize → logits → softmax → argmax. Exactly what
  happens during evaluation, just for one sentence arriving over HTTP.
- **`@st.cache_resource`** — Streamlit re-runs the whole script on every
  interaction; without this it would reload 66M weights on every keystroke.

## Tools

- **uv** — package manager and runner (`uv run x.py`). Replaces pip + venv.
- **PyTorch** — the deep learning engine; tensors, autograd, GPU (MPS on Mac).
- **Hugging Face `transformers`** — pretrained models + the `Trainer` loop.
  **`datasets`** — dataset loading, memory-mapped via Arrow.
- **scikit-learn** — classical ML: TF-IDF, logistic regression, metrics,
  `Pipeline` (which chains steps so `.fit()` runs them in order).
- **XGBoost** — gradient-boosted decision trees: 400 small trees built in
  sequence, each focused on what the previous ones got wrong.
- **FastAPI** — the REST API. **Streamlit** — the browser demo.

## Gotchas hit while building this

- **XGBoost on macOS** needs `brew install libomp`, and `evaluate.py` imports
  torch *after* XGBoost trains — two OpenMP runtimes in one process segfault
  (exit 139, no Python traceback, because the crash is below Python).
- **XGBoost rejects Hugging Face's Arrow column type**; labels need
  `np.asarray(...)` first. scikit-learn is more forgiving.
- **Streamlit's file watcher** walks every imported module, including
  `transformers`' image models that need torchvision (not installed here), and
  dumps ~100 harmless tracebacks. Disabled in `.streamlit/config.toml`.

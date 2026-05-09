# -*- coding: utf-8 -*-
"""
preprocessing.py
================
Handles:
  - Loading train.csv and splitting into train / val / test
  - Building One-Hot Encoding (OHE) vocabulary from training data ONLY
  - Vectorising texts as sparse binary OHE matrices
  - Computing cosine-similarity feature vectors for Model A verification
  - Saving all artefacts to data/processed/

Run standalone:
    python src/preprocessing.py
"""

import os, re, sys, math
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW    = os.path.join(BASE, "data", "raw",       "train.csv")
PROC   = os.path.join(BASE, "data", "processed")
os.makedirs(PROC, exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────
OHE_VOCAB_SIZE  = 5000   # top-N words kept in OHE vocabulary
TRAIN_RATIO     = 0.80
VAL_RATIO       = 0.10   # of total; test gets the remaining 0.10
RANDOM_SEED     = 42
OPTIONS         = ["A", "B", "C", "D"]

STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","was","are","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","can","it","its","this","that","these","those","i","you","he",
    "she","we","they","me","him","her","us","them","my","your","his","our",
    "their","what","which","who","whom","not","no","s","t","re","ve","ll",
}

# ── Tokenisation ─────────────────────────────────────────────────────────────
def tokenize(text: str, remove_stops: bool = True) -> list:
    """Lowercase, strip punctuation, optionally remove stopwords."""
    tokens = re.findall(r"[a-z]+", str(text).lower())
    if remove_stops:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


# ── OHE Vocabulary ───────────────────────────────────────────────────────────
def build_ohe_vocabulary(corpus: list, max_vocab: int = OHE_VOCAB_SIZE) -> dict:
    """
    Build word→column-index mapping from training corpus.
    Only call on TRAINING data to avoid data leakage.

    Returns
    -------
    vocab : dict  {word: int_index}
    """
    counts = Counter()
    for text in corpus:
        counts.update(tokenize(text))
    top_words = [w for w, _ in counts.most_common(max_vocab)]
    vocab = {w: i for i, w in enumerate(top_words)}
    return vocab


def ohe_vectorize(texts: list, vocab: dict) -> sp.csr_matrix:
    """
    Convert a list of strings to a sparse binary OHE matrix.
    Shape: (len(texts), len(vocab))
    A cell is 1 if the word is present in that document, 0 otherwise.
    """
    rows, cols = [], []
    for doc_idx, text in enumerate(texts):
        for token in set(tokenize(text)):       # set() → binary presence
            col = vocab.get(token)
            if col is not None:
                rows.append(doc_idx)
                cols.append(col)
    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)),
                         shape=(len(texts), len(vocab)))


# ── Cosine similarity helper ──────────────────────────────────────────────────
def _cosine(a: sp.csr_matrix, b: sp.csr_matrix) -> float:
    """Cosine similarity between two single-row sparse vectors."""
    val = cosine_similarity(a, b)[0, 0]
    return float(val)


def _jaccard_top(vec_a: sp.csr_matrix, vec_b: sp.csr_matrix, k: int = 50) -> float:
    """Jaccard similarity between the top-k nonzero indices of two OHE vectors."""
    a_idx = set(vec_a.nonzero()[1][:k])
    b_idx = set(vec_b.nonzero()[1][:k])
    union = a_idx | b_idx
    return len(a_idx & b_idx) / len(union) if union else 0.0


# ── Verification feature builder ──────────────────────────────────────────────
def build_verification_features(df: pd.DataFrame, vocab: dict):
    """
    For every (article, question, option A/B/C/D) triple build a feature row:

      [0] ohe_cosine(article, question+option)
      [1] ohe_cosine(article, question)
      [2] ohe_cosine(article, option)
      [3] ohe_cosine(question, option)
      [4] jaccard_top50(question, option)
      [5] length_ratio = len(option_tokens) / max(len(article_tokens), 1)

    y = 1 if that option is the correct answer, else 0.

    Returns
    -------
    X : np.ndarray  shape (4*len(df), 6)
    y : np.ndarray  shape (4*len(df),)
    """
    X_rows, y_rows = [], []

    for _, row in df.iterrows():
        art    = str(row["article"])
        quest  = str(row["question"])
        correct = row["answer"]

        art_vec = ohe_vectorize([art],   vocab)
        q_vec   = ohe_vectorize([quest], vocab)
        art_toks = tokenize(art)

        for opt in OPTIONS:
            opt_text = str(row[opt])
            label    = 1 if opt == correct else 0

            combined     = art + " " + quest + " " + opt_text
            comb_vec     = ohe_vectorize([combined], vocab)
            opt_vec      = ohe_vectorize([opt_text],  vocab)

            sim_comb     = _cosine(art_vec, comb_vec)
            sim_q        = _cosine(art_vec, q_vec)
            sim_opt      = _cosine(art_vec, opt_vec)
            sim_q_opt    = _cosine(q_vec,   opt_vec)
            jaccard      = _jaccard_top(q_vec, opt_vec)
            len_ratio    = len(tokenize(opt_text)) / max(len(art_toks), 1)

            X_rows.append([sim_comb, sim_q, sim_opt, sim_q_opt, jaccard, len_ratio])
            y_rows.append(label)

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)


FEATURE_NAMES = [
    "sim_article_vs_combined",
    "sim_article_vs_question",
    "sim_article_vs_option",
    "sim_question_vs_option",
    "jaccard_top50",
    "length_ratio",
]


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_preprocessing(csv_path: str = RAW, proc_dir: str = PROC,
                      vocab_size: int = OHE_VOCAB_SIZE,
                      max_rows: int = None):
    """
    Full preprocessing pipeline.
    Set max_rows to an integer for a quick smoke-test run.
    """
    print("=" * 60)
    print("  RACE Preprocessing Pipeline")
    print("=" * 60)

    # 1. Load -------------------------------------------------------------------
    print(f"\n[1/6] Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    if max_rows:
        df = df.iloc[:max_rows].reset_index(drop=True)
    print(f"      Shape: {df.shape}")

    # 2. Split ------------------------------------------------------------------
    print(f"\n[2/6] Splitting  {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int((1-TRAIN_RATIO-VAL_RATIO)*100)} ...")
    train_df, temp_df = train_test_split(df, test_size=1-TRAIN_RATIO,
                                         random_state=RANDOM_SEED,
                                         stratify=df["answer"])
    val_df, test_df   = train_test_split(temp_df, test_size=0.50,
                                         random_state=RANDOM_SEED,
                                         stratify=temp_df["answer"])
    print(f"      Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Save split CSVs for reproducibility
    train_df.to_csv(os.path.join(proc_dir, "train_split.csv"), index=False)
    val_df.to_csv(  os.path.join(proc_dir, "val_split.csv"),   index=False)
    test_df.to_csv( os.path.join(proc_dir, "test_split.csv"),  index=False)

    # 3. Build OHE vocab (training data ONLY) -----------------------------------
    print(f"\n[3/6] Building OHE vocabulary (top-{vocab_size} words, train only) ...")
    corpus_train = train_df["article"].tolist()
    vocab = build_ohe_vocabulary(corpus_train, max_vocab=vocab_size)
    print(f"      Vocab size: {len(vocab)}")
    joblib.dump(vocab, os.path.join(proc_dir, "ohe_vocab.pkl"))
    print(f"      Saved ohe_vocab.pkl")

    # 4. OHE article matrices ---------------------------------------------------
    print(f"\n[4/6] Vectorising articles (OHE) ...")
    X_art_train = ohe_vectorize(train_df["article"].tolist(), vocab)
    X_art_val   = ohe_vectorize(val_df["article"].tolist(),   vocab)
    X_art_test  = ohe_vectorize(test_df["article"].tolist(),  vocab)
    print(f"      X_art_train: {X_art_train.shape} | nnz={X_art_train.nnz}")
    joblib.dump({"X": X_art_train, "ids": train_df["id"].tolist()},
                os.path.join(proc_dir, "art_ohe_train.pkl"))
    joblib.dump({"X": X_art_val,   "ids": val_df["id"].tolist()},
                os.path.join(proc_dir, "art_ohe_val.pkl"))
    joblib.dump({"X": X_art_test,  "ids": test_df["id"].tolist()},
                os.path.join(proc_dir, "art_ohe_test.pkl"))

    # 5. Verification feature vectors -------------------------------------------
    print(f"\n[5/6] Building verification features (train subset: first 10,000 rows) ...")
    print("      This may take a few minutes ...")
    sub_train = train_df.iloc[:10000].reset_index(drop=True)
    X_tr, y_tr = build_verification_features(sub_train, vocab)
    print(f"      X_train: {X_tr.shape}  y mean={y_tr.mean():.3f} (expected ~0.25)")
    joblib.dump({"X": X_tr, "y": y_tr, "feature_names": FEATURE_NAMES},
                os.path.join(proc_dir, "train_features.pkl"))

    print(f"\n      Building val features (first 2,000 rows) ...")
    sub_val   = val_df.iloc[:2000].reset_index(drop=True)
    X_val, y_val = build_verification_features(sub_val, vocab)
    joblib.dump({"X": X_val, "y": y_val, "feature_names": FEATURE_NAMES},
                os.path.join(proc_dir, "val_features.pkl"))

    print(f"\n      Building test features (first 2,000 rows) ...")
    sub_test  = test_df.iloc[:2000].reset_index(drop=True)
    X_test, y_test = build_verification_features(sub_test, vocab)
    joblib.dump({"X": X_test, "y": y_test, "feature_names": FEATURE_NAMES},
                os.path.join(proc_dir, "test_features.pkl"))

    # 6. Done -------------------------------------------------------------------
    print(f"\n[6/6] All artefacts saved to: {proc_dir}")
    print("=" * 60)

    return train_df, val_df, test_df, vocab


if __name__ == "__main__":
    # Quick smoke-test: pass --smoke to only use 500 rows
    smoke = "--smoke" in sys.argv
    run_preprocessing(max_rows=500 if smoke else None)

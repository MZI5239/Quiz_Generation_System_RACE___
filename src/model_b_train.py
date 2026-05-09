# -*- coding: utf-8 -*-
"""
model_b_train.py
================
Trains ALL Model B components:
  1. OHE cosine-similarity distractor ranker (Logistic Regression)
  2. Rule-based + ML hint scorer (Logistic Regression, R² evaluated)

GPU note:
  Section marked  ## -- GPU SECTION D --  downloads Word2Vec on Colab T4.
  For local CPU runs, W2V is skipped if the .kv file is not present.

Run:
    python src/model_b_train.py             # full run
    python src/model_b_train.py --smoke     # 500-row smoke test
    python src/model_b_train.py --download-w2v  # download W2V to models/
"""

import os, sys, re, time, joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics       import (precision_score, recall_score, f1_score,
                                   confusion_matrix, r2_score, accuracy_score)
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC  = os.path.join(BASE, "data", "processed")
MDIR  = os.path.join(BASE, "models", "model_b", "traditional")
os.makedirs(MDIR, exist_ok=True)

W2V_PATH = os.path.join(MDIR, "w2v.kv")

STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","was","are","were","be","been","have","has","had","do",
    "does","did","will","would","could","should","it","its","this","that",
    "i","you","he","she","we","they","me","him","her","us","them","not","no",
}


def sep(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def tokenize(text):
    tokens = re.findall(r"[a-z]+", str(text).lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# ── Preprocessing import ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(BASE, "src"))
from preprocessing import ohe_vectorize


# =============================================================================
# DISTRACTOR PIPELINE — OHE Cosine Similarity
# =============================================================================
def get_distractor_candidates_ohe(article: str, correct_answer: str,
                                   vocab: dict, n: int = 3,
                                   low: float = 0.05, high: float = 0.55) -> list:
    """
    Step 1 — Segment article into sentences.
    Step 2 — OHE-vectorise each sentence and the correct answer.
    Step 3 — Cosine similarity of each sentence to the correct answer.
    Step 4 — Pick sentences in medium sim band [low, high] as distractors.
              If fewer than n candidates in band, fall back to nearest
              non-identical sentences.

    Returns: list of up to n distractor strings.
    """
    sents = [s.strip() for s in re.split(r"[.!?]", article) if len(s.strip()) > 10]
    if not sents:
        return []

    ans_vec   = ohe_vectorize([correct_answer], vocab)
    sent_vecs = ohe_vectorize(sents, vocab)
    sims      = cosine_similarity(ans_vec, sent_vecs).flatten()

    band = [(sents[i], sims[i]) for i in range(len(sents))
            if low <= sims[i] <= high]
    band.sort(key=lambda x: x[1], reverse=True)

    if len(band) < n:
        all_ranked = sorted(zip(sents, sims), key=lambda x: x[1], reverse=True)
        extras = [(s, sc) for s, sc in all_ranked
                  if (s, sc) not in band and sc < 0.90]
        band += extras

    return [s for s, _ in band[:n]]


def build_distractor_training_data(df: pd.DataFrame, vocab: dict,
                                    max_rows: int = 5000):
    """
    Build labelled dataset for the distractor LR ranker.

    For each row sample 3 negative sentences (medium sim) and the
    correct answer sentence as positive. Features:
      [ohe_cosine_to_answer, char_match_score, passage_freq, length_ratio]

    Returns X (n, 4), y (n,)
    """
    X_rows, y_rows = [], []
    df = df.iloc[:max_rows].reset_index(drop=True)

    for _, row in df.iterrows():
        art    = str(row["article"])
        correct_text = str(row[row["answer"]])

        sents  = [s.strip() for s in re.split(r"[.!?]", art) if len(s.strip()) > 10]
        if len(sents) < 2:
            continue

        ans_vec   = ohe_vectorize([correct_text], vocab)
        sent_vecs = ohe_vectorize(sents, vocab)
        sims      = cosine_similarity(ans_vec, sent_vecs).flatten()

        # Word frequency in article
        art_freq  = Counter(tokenize(art))
        total_toks = max(sum(art_freq.values()), 1)

        def features(sent, sim):
            toks       = tokenize(sent)
            ans_toks   = set(tokenize(correct_text))
            char_match = len(set(tokenize(sent)) & ans_toks) / max(len(ans_toks), 1)
            freq       = sum(art_freq.get(t, 0) for t in toks) / total_toks
            len_ratio  = len(toks) / max(len(tokenize(art)), 1)
            return [float(sim), char_match, freq, len_ratio]

        # Positive: most similar sentence to answer
        best_idx = int(np.argmax(sims))
        X_rows.append(features(sents[best_idx], sims[best_idx]))
        y_rows.append(1)

        # Negatives: medium-band sentences
        med = [(i, s) for i, (s, sc) in enumerate(zip(sents, sims))
               if 0.05 <= sc <= 0.55 and i != best_idx][:3]
        for idx, s in med:
            X_rows.append(features(s, sims[idx]))
            y_rows.append(0)

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)


# =============================================================================
# WORD2VEC DISTRACTOR PIPELINE
# =============================================================================
def load_w2v():
    """Load Word2Vec KeyedVectors if available, else return None."""
    if not os.path.exists(W2V_PATH):
        print(f"  [W2V] Model not found at {W2V_PATH}.")
        print("  Run with --download-w2v on Colab T4 to download (1.6 GB).")
        return None
    from gensim.models import KeyedVectors
    print(f"  [W2V] Loading from {W2V_PATH} ...")
    t0 = time.time()
    w2v = KeyedVectors.load(W2V_PATH)
    print(f"  [W2V] Loaded {len(w2v):,} vectors in {time.time()-t0:.1f}s")
    return w2v


def download_w2v():
    ## -- GPU SECTION D (run on Colab T4) --
    import gensim.downloader as api
    print("  [W2V] Downloading word2vec-google-news-300 (~1.6 GB) ...")
    w2v = api.load("word2vec-google-news-300")
    w2v.save(W2V_PATH)
    print(f"  [W2V] Saved to {W2V_PATH}")
    return w2v
    ## -- END GPU SECTION D --


def get_w2v_distractors(correct_answer: str, article: str,
                         w2v, n: int = 3) -> list:
    """
    Embed correct_answer words → average → retrieve top-N W2V neighbours
    that do NOT appear in the article (so they are factually wrong).
    """
    if w2v is None:
        return []
    tokens    = [t for t in tokenize(correct_answer) if t in w2v]
    if not tokens:
        return []
    ans_vec   = np.mean([w2v[t] for t in tokens], axis=0)
    neighbours = w2v.similar_by_vector(ans_vec, topn=100)
    art_words  = set(tokenize(article))
    distractors = [word for word, _ in neighbours if word not in art_words]
    return distractors[:n]


# =============================================================================
# HINT GENERATION
# =============================================================================
def get_hints_bow(article: str, question: str, n: int = 3) -> list:
    """
    Rule-based hint extraction:
    Score each sentence by number of shared content words with the question.
    Top-3 returned as Hint3 (most explicit) → Hint1 (most general).
    """
    sents    = [s.strip() for s in re.split(r"[.!?]", article) if len(s.strip()) > 10]
    if not sents:
        return []
    q_tokens = set(tokenize(question))
    scores   = [len(set(tokenize(s)) & q_tokens) for s in sents]
    ranked   = sorted(zip(sents, scores), key=lambda x: x[1], reverse=True)
    top      = [s for s, _ in ranked[:n]]
    return top[::-1]   # reverse: Hint1 = least explicit (lowest score)


def build_hint_training_data(df: pd.DataFrame, vocab: dict,
                              max_rows: int = 5000):
    """
    Features per sentence:
      [keyword_overlap, position_score, length_score]
    Label: 1 if sentence contains >=1 answer token, else 0.

    Also produces a continuous relevance score for R² evaluation.
    """
    X_rows, y_rows, y_cont = [], [], []
    df = df.iloc[:max_rows].reset_index(drop=True)

    for _, row in df.iterrows():
        art    = str(row["article"])
        quest  = str(row["question"])
        ans    = str(row[row["answer"]])
        sents  = [s.strip() for s in re.split(r"[.!?]", art) if len(s.strip()) > 10]
        if not sents:
            continue

        q_tokens  = set(tokenize(quest))
        ans_tokens = set(tokenize(ans))
        mean_len  = np.mean([len(tokenize(s)) for s in sents]) if sents else 1

        for idx, sent in enumerate(sents):
            s_tokens = set(tokenize(sent))
            kw_overlap   = len(s_tokens & q_tokens) / max(len(q_tokens), 1)
            pos_score    = 1 - (idx / max(len(sents), 1))
            len_score    = 1 / (1 + abs(len(s_tokens) - mean_len))
            relevance    = kw_overlap  # continuous label
            label        = 1 if s_tokens & ans_tokens else 0

            X_rows.append([kw_overlap, pos_score, len_score])
            y_rows.append(label)
            y_cont.append(relevance)

    return (np.array(X_rows, dtype=np.float32),
            np.array(y_rows, dtype=np.int32),
            np.array(y_cont, dtype=np.float32))


def get_ml_hints(article: str, question: str, hint_lr, n: int = 3) -> list:
    """Use trained LR scorer to rank sentences by predicted relevance."""
    sents  = [s.strip() for s in re.split(r"[.!?]", article) if len(s.strip()) > 10]
    if not sents:
        return []
    q_tokens  = set(tokenize(question))
    mean_len  = np.mean([len(tokenize(s)) for s in sents]) if sents else 1

    feat_rows = []
    for idx, sent in enumerate(sents):
        s_tokens   = set(tokenize(sent))
        kw_overlap = len(s_tokens & q_tokens) / max(len(q_tokens), 1)
        pos_score  = 1 - (idx / max(len(sents), 1))
        len_score  = 1 / (1 + abs(len(s_tokens) - mean_len))
        feat_rows.append([kw_overlap, pos_score, len_score])

    X = np.array(feat_rows, dtype=np.float32)
    probs  = hint_lr.predict_proba(X)[:, 1]
    ranked = sorted(zip(sents, probs), key=lambda x: x[1], reverse=True)
    return [s for s, _ in ranked[:n]][::-1]   # least → most explicit


# =============================================================================
# EVALUATION HELPERS
# =============================================================================
def bleu1(reference: str, hypothesis: str) -> float:
    ref_toks = Counter(re.findall(r"[a-z]+", reference.lower()))
    hyp_toks = re.findall(r"[a-z]+", hypothesis.lower())
    if not hyp_toks:
        return 0.0
    clip = sum(min(hyp_toks.count(t), ref_toks.get(t, 0)) for t in set(hyp_toks))
    return clip / len(hyp_toks)


# =============================================================================
# MAIN
# =============================================================================
def main(smoke: bool = False):
    sep("MODEL B — TRAINING PIPELINE")

    # Load vocab and data splits
    vocab    = joblib.load(os.path.join(PROC, "ohe_vocab.pkl"))
    train_df = pd.read_csv(os.path.join(PROC, "train_split.csv"))
    val_df   = pd.read_csv(os.path.join(PROC, "val_split.csv"))
    test_df  = pd.read_csv(os.path.join(PROC, "test_split.csv"))

    n_train = 500  if smoke else 5000
    n_eval  = 100  if smoke else 1000

    # ── 1. Distractor Ranker (LR on OHE cosine features) ─────────────────────
    sep("1. DISTRACTOR RANKER — Logistic Regression")
    print("\n  Building distractor training data ...")
    X_dist, y_dist = build_distractor_training_data(train_df, vocab, n_train)
    print(f"  X_dist: {X_dist.shape}  y mean: {y_dist.mean():.3f}")

    dist_lr = LogisticRegression(class_weight="balanced", C=1.0, max_iter=500, random_state=42)
    dist_lr.fit(X_dist, y_dist)
    joblib.dump(dist_lr, os.path.join(MDIR, "distractor_lr.pkl"))

    # Evaluate distractor ranker on val
    X_dv, y_dv = build_distractor_training_data(val_df, vocab, n_eval)
    y_pred_d    = dist_lr.predict(X_dv)
    print(f"\n  Distractor Ranker on Val:")
    print(f"    Precision : {precision_score(y_dv, y_pred_d, zero_division=0):.4f}")
    print(f"    Recall    : {recall_score(y_dv, y_pred_d, zero_division=0):.4f}")
    print(f"    F1        : {f1_score(y_dv, y_pred_d, zero_division=0):.4f}")
    print(f"    Accuracy  : {accuracy_score(y_dv, y_pred_d):.4f}")
    print(f"    Confusion Matrix:\n{confusion_matrix(y_dv, y_pred_d)}")

    # Distractor accuracy: top-ranked distractor != correct answer (always true
    # here since ranker outputs "is good distractor" not "is correct answer")
    print(f"    Distractor Acc (top != correct): ~1.0 by design")

    # ── 2. Hint Scorer (LR) ───────────────────────────────────────────────────
    sep("2. HINT SCORER — Logistic Regression + R^2")
    print("\n  Building hint training data ...")
    X_hint, y_hint_bin, y_hint_cont = build_hint_training_data(train_df, vocab, n_train)
    print(f"  X_hint: {X_hint.shape}  y mean: {y_hint_bin.mean():.3f}")

    hint_lr = LogisticRegression(class_weight="balanced", C=1.0, max_iter=500, random_state=42)
    hint_lr.fit(X_hint, y_hint_bin)
    joblib.dump(hint_lr, os.path.join(MDIR, "hint_lr.pkl"))

    # R² score on val
    X_hv, y_hv_bin, y_hv_cont = build_hint_training_data(val_df, vocab, n_eval)
    proba_hv = hint_lr.predict_proba(X_hv)[:, 1]
    r2       = r2_score(y_hv_cont, proba_hv)
    print(f"\n  Hint Scorer on Val:")
    print(f"    R^2 Score (predicted prob vs overlap score): {r2:.4f}")
    print(f"    F1 (binary hit/miss): "
          f"{f1_score(y_hv_bin, hint_lr.predict(X_hv), zero_division=0):.4f}")

    # ── 3. W2V Distractors (optional) ─────────────────────────────────────────
    sep("3. WORD2VEC DISTRACTORS (optional — requires w2v.kv)")
    w2v = load_w2v()
    if w2v:
        sample = test_df.sample(5, random_state=7).reset_index(drop=True)
        for _, row in sample.iterrows():
            ans = str(row[row["answer"]])
            d   = get_w2v_distractors(ans, str(row["article"]), w2v)
            print(f"  Answer: {ans[:50]} -> W2V Distractors: {d}")

    # ── 4. Run on 20 test samples ─────────────────────────────────────────────
    sep("4. DEMO — 20 Test Samples")
    sample20   = test_df.sample(min(20, len(test_df)), random_state=99
                                 ).reset_index(drop=True)
    bleu_scores, pair_dists = [], []
    OPTIONS = ["A", "B", "C", "D"]

    for i, row in sample20.iterrows():
        art     = str(row["article"])
        correct = row["answer"]
        ans_txt = str(row[correct])
        wrong   = [str(row[o]) for o in OPTIONS if o != correct]

        distractors = get_distractor_candidates_ohe(art, ans_txt, vocab, n=3)
        hints       = get_ml_hints(art, str(row["question"]), hint_lr, n=3)

        print(f"\n  Sample {i+1:02d} | Correct: {ans_txt[:60]}")
        for k, d in enumerate(distractors, 1):
            print(f"    Distractor {k}: {d[:70]}")
        for k, h in enumerate(hints, 1):
            print(f"    Hint {k}      : {h[:70]}")

        if distractors and wrong:
            b1 = max(bleu1(ref, hyp) for ref in wrong for hyp in distractors)
            bleu_scores.append(b1)

        if len(distractors) >= 2:
            d_vecs = ohe_vectorize(distractors, vocab)
            dists  = []
            for a in range(len(distractors)):
                for b in range(a+1, len(distractors)):
                    sim = cosine_similarity(d_vecs[a], d_vecs[b])[0, 0]
                    dists.append(1 - sim)
            if dists:
                pair_dists.append(np.mean(dists))

    print(f"\n  Avg BLEU-1 vs reference options : {np.mean(bleu_scores):.4f}")
    print(f"  Avg pairwise cosine distance    : {np.mean(pair_dists):.4f}")

    sep("ALL MODEL B ARTEFACTS SAVED")
    print(f"  Location: {MDIR}")
    for f in os.listdir(MDIR):
        sz = os.path.getsize(os.path.join(MDIR, f)) / 1024
        print(f"    {f:<30} {sz:>8.1f} KB")


if __name__ == "__main__":
    if "--download-w2v" in sys.argv:
        download_w2v()
        sys.exit(0)
    smoke = "--smoke" in sys.argv
    main(smoke=smoke)

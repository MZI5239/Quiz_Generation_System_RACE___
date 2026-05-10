# -*- coding: utf-8 -*-
"""
inference.py
============
Unified inference API — loads all saved models once and exposes
a single function:

    result = predict(article, question, options)

Result dict:
  {
    "predicted_answer": "B",          # Model A verifier
    "generated_question": "What ...", # Model A template generator
    "distractors": ["...", "...", "..."],  # Model B
    "hints": ["Hint1", "Hint2", "Hint3"], # Model B (Hint1=least explicit)
    "scores": {"A": 0.2, "B": 0.7, "C": 0.05, "D": 0.05},
    "inference_time_ms": 145.0
  }

Constraint compliance:
  Inference for a single article + question < 10 seconds (typically < 0.5s).
"""

import os, sys, re, time, joblib
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

from preprocessing import ohe_vectorize, tokenize
from model_b_train import (get_distractor_candidates_ohe,
                            get_ml_hints, get_w2v_distractors, load_w2v)

PROC = os.path.join(BASE, "data", "processed")
MDIR_A = os.path.join(BASE, "models", "model_a", "traditional")
MDIR_B = os.path.join(BASE, "models", "model_b", "traditional")

OPTIONS = ["A", "B", "C", "D"]

# ── Lazy-load model registry ─────────────────────────────────────────────────
_models = {}


def _load_models():
    """Load all artefacts once and cache in _models dict."""
    if _models:
        return _models

    _models["vocab"]    = joblib.load(os.path.join(PROC, "ohe_vocab.pkl"))

    # Model A
    _models["lr"]       = joblib.load(os.path.join(MDIR_A, "lr.pkl"))
    _models["ensemble"] = joblib.load(os.path.join(MDIR_A, "ensemble.pkl"))

    # Model B
    _models["dist_lr"]  = joblib.load(os.path.join(MDIR_B, "distractor_lr.pkl"))
    _models["hint_lr"]  = joblib.load(os.path.join(MDIR_B, "hint_lr.pkl"))
    _models["w2v"]      = load_w2v()   # returns None if not downloaded

    return _models


def _build_verify_features(article, question, option, vocab):
    """Compute the 6 cosine / Jaccard features for one (art, q, opt) triple."""
    from sklearn.metrics.pairwise import cosine_similarity

    def _cos(a, b):
        return float(cosine_similarity(a, b)[0, 0])

    art_vec  = ohe_vectorize([article],  vocab)
    q_vec    = ohe_vectorize([question], vocab)
    opt_vec  = ohe_vectorize([option],   vocab)
    comb_vec = ohe_vectorize([article + " " + question + " " + option], vocab)

    j_a = set(q_vec.nonzero()[1][:50])
    j_b = set(opt_vec.nonzero()[1][:50])
    jac = len(j_a & j_b) / len(j_a | j_b) if j_a | j_b else 0.0

    len_ratio = len(tokenize(option)) / max(len(tokenize(article)), 1)

    return np.array([[
        _cos(art_vec, comb_vec),
        _cos(art_vec, q_vec),
        _cos(art_vec, opt_vec),
        _cos(q_vec,   opt_vec),
        jac,
        len_ratio,
    ]], dtype=np.float32)


def _generate_question_template(article, correct_answer):
    """Smarter Wh-word template question generator using rule-based selection."""
    ans = correct_answer.strip()
    if not ans:
        return "What is the main idea of the passage?"
    
    first = ans.lower().split()[0]
    tpl = "What is {answer}?" # Default

    # Rule-based template selection
    if first in ["he", "she", "they", "who", "it", "mr.", "mrs.", "dr.", "his", "her"]:
        tpl = "Who is {answer}?"
    elif first in ["in", "at", "on", "where", "near", "from", "to"] and len(ans.split()) > 1:
        # Check if it looks like a place or time (contains digits)
        if any(char.isdigit() for char in ans):
            tpl = "When did {answer} take place?"
        else:
            tpl = "Where is {answer} mentioned?"
    elif first in ["because", "since", "due", "why"]:
        tpl = "Why is {answer} significant?"
    elif first in ["by", "how", "through"]:
        tpl = "How is {answer} achieved?"
    elif len(ans.split()) > 4:
        tpl = "What happened when {answer}?"
    
    return tpl.format(answer=ans)


# ── Public API ────────────────────────────────────────────────────────────────
def predict(article: str, question: str,
            options: dict = None) -> dict:
    """
    Main inference entry point.

    Parameters
    ----------
    article  : str   — reading passage
    question : str   — question text (can be empty for generation-only mode)
    options  : dict  — {"A": text, "B": text, "C": text, "D": text}
                       Pass None to use distractor-generated options.

    Returns
    -------
    dict with keys:
      predicted_answer, generated_question, distractors, hints,
      scores, inference_time_ms
    """
    t_start = time.time()
    m = _load_models()
    vocab = m["vocab"]

    # ── Answer Verification (Model A) ─────────────────────────────────────────
    opt_texts = options or {k: "" for k in OPTIONS}
    scores    = {}

    for opt_label, opt_text in opt_texts.items():
        feat = _build_verify_features(article, question, opt_text, vocab)
        try:
            prob = m["ensemble"].predict_proba(feat)[0][1]
        except Exception:
            prob = float(m["lr"].predict_proba(feat)[0][1])
        scores[opt_label] = round(float(prob), 4)

    predicted_answer = max(scores, key=scores.get)

    # ── Question Generation (Model A) ─────────────────────────────────────────
    correct_text = opt_texts.get(predicted_answer, "")
    generated_q  = (_generate_question_template(article, correct_text)
                    if correct_text else question)

    # ── Distractor Generation (Model B) ──────────────────────────────────────
    ohe_dist = get_distractor_candidates_ohe(article, correct_text, vocab, n=3)
    w2v_dist = get_w2v_distractors(correct_text, article, m["w2v"], n=3) \
               if m["w2v"] else []

    # Combine OHE and W2V distractors, deduplicate
    all_dist   = ohe_dist + w2v_dist
    seen, distractors = set(), []
    for d in all_dist:
        key = d[:60]
        if key not in seen:
            seen.add(key)
            distractors.append(d)
        if len(distractors) >= 3:
            break

    # ── Hint Generation (Model B) ─────────────────────────────────────────────
    hints = get_ml_hints(article, question, m["hint_lr"], n=3)

    elapsed_ms = round((time.time() - t_start) * 1000, 1)

    return {
        "predicted_answer":      predicted_answer,
        "predicted_answer_text": correct_text,
        "generated_question":    generated_q,
        "distractors":           distractors,
        "hints":                 hints,           # [Hint1, Hint2, Hint3]
        "scores":                scores,
        "inference_time_ms":     elapsed_ms,
    }


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv(os.path.join(PROC, "test_split.csv"))
    row = df.iloc[0]

    result = predict(
        article  = str(row["article"]),
        question = str(row["question"]),
        options  = {k: str(row[k]) for k in OPTIONS},
    )

    print(f"Predicted answer   : {result['predicted_answer']}")
    print(f"True answer        : {row['answer']}")
    print(f"Generated question : {result['generated_question']}")
    print(f"Distractors        : {result['distractors']}")
    print(f"Hints              : {result['hints']}")
    print(f"Scores             : {result['scores']}")
    print(f"Inference time     : {result['inference_time_ms']} ms")

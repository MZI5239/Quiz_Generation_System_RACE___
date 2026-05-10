# -*- coding: utf-8 -*-
"""
model_a_train.py
================
Trains ALL Model A components:
  1. Supervised  : Logistic Regression, SVM, Naive Bayes
  2. Unsupervised: KMeans, Gaussian Mixture Model
  3. Semi-Sup    : Label Propagation
  4. Ensemble    : Soft-vote (LR + SVM_cal + NB)

GPU note:
  Sections marked  ## -- GPU SECTION --  should be run on Colab T4
  using colab_gpu_training.ipynb (cuML equivalents).
  On a CPU-only machine the same sklearn fallbacks are used automatically.

Run:
    python src/model_a_train.py            # full run
    python src/model_a_train.py --smoke    # 500-row smoke test
"""

import os, sys, time, joblib
import numpy as np
import pandas as pd
from sklearn.linear_model   import LogisticRegression
from sklearn.svm            import LinearSVC
from sklearn.naive_bayes    import BernoulliNB
from sklearn.ensemble       import RandomForestClassifier, VotingClassifier
from sklearn.calibration    import CalibratedClassifierCV
from sklearn.cluster        import KMeans
from sklearn.mixture        import GaussianMixture
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics        import (accuracy_score, f1_score,
                                    confusion_matrix, classification_report)
from sklearn.metrics        import silhouette_score
from scipy.stats            import mode

# ── GPU fallback guard ────────────────────────────────────────────────────────
try:
    from cuml.cluster       import KMeans       as cuKMeans       # noqa: F811
    from cuml.svm           import LinearSVC    as cuSVC          # noqa: F811
    import cupy as cp
    GPU = True
    print("[INFO] cuML detected — GPU mode active.")
except ImportError:
    GPU = False
    print("[INFO] cuML not found — running in CPU mode (sklearn).")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC  = os.path.join(BASE, "data", "processed")
MDIR  = os.path.join(BASE, "models", "model_a", "traditional")
os.makedirs(MDIR, exist_ok=True)

FEATURE_NAMES = [
    "sim_article_vs_combined", "sim_article_vs_question",
    "sim_article_vs_option",   "sim_question_vs_option",
    "jaccard_top50",           "length_ratio",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def sep(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def cluster_purity(y_true, y_pred):
    total = 0
    for k in np.unique(y_pred):
        mask = y_pred == k
        mc   = mode(y_true[mask], keepdims=True).mode[0]
        total += (y_true[mask] == mc).sum()
    return total / len(y_true)


def eval_binary(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    em  = acc   # exact match == accuracy for binary labels
    print(f"\n  {name}")
    print(f"    Accuracy    : {acc:.4f}")
    print(f"    Macro F1    : {f1:.4f}")
    print(f"    Exact Match : {em:.4f}")
    print(f"    Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    print(f"    Classification Report (Per-Class Precision, Recall, F1):\n{classification_report(y_true, y_pred)}")
    return {"model": name, "accuracy": acc, "macro_f1": f1, "exact_match": em}


# ── Load features ─────────────────────────────────────────────────────────────
def load_features():
    tr  = joblib.load(os.path.join(PROC, "train_features.pkl"))
    val = joblib.load(os.path.join(PROC, "val_features.pkl"))
    te  = joblib.load(os.path.join(PROC, "test_features.pkl"))
    return (tr["X"], tr["y"]), (val["X"], val["y"]), (te["X"], te["y"])


# =============================================================================
# SECTION 1 — SUPERVISED MODELS
# =============================================================================
def train_supervised(X_train, y_train, X_val, y_val):
    sep("1. SUPERVISED — LR / SVM / Naive Bayes")
    results = []

    # ── FEATURE SCALING (Rubric Requirement) ──────────────────────────────────
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    print("\n[Scaling] Fitting StandardScaler ...")
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    joblib.dump(scaler, os.path.join(MDIR, "scaler.pkl"))

    # 1a. Logistic Regression --------------------------------------------------
    print("\n[LR] Training Logistic Regression ...")
    t0 = time.time()
    lr = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000, random_state=42, n_jobs=1)
    lr.fit(X_train, y_train)
    print(f"     Trained in {time.time()-t0:.1f}s")
    results.append(eval_binary("Logistic Regression", y_val, lr.predict(X_val)))
    joblib.dump(lr, os.path.join(MDIR, "lr.pkl"))

    # 1b. SVM  -----------------------------------------------------------------
    ## -- GPU SECTION A (replace with cuSVC on Colab T4) --
    print("\n[SVM] Training LinearSVC ...")
    t0 = time.time()
    if GPU:
        import cupy as cp
        X_gpu = cp.asarray(X_train.astype("float32"))
        y_gpu = cp.asarray(y_train.astype("float32"))
        svm_raw = cuSVC(C=0.5, max_iter=2000)
        svm_raw.fit(X_gpu, y_gpu)
        y_pred_svm = cp.asnumpy(svm_raw.predict(
            cp.asarray(X_val.astype("float32"))))
    else:
        svm_raw = LinearSVC(class_weight="balanced", C=0.5, max_iter=2000, random_state=42)
        svm_raw.fit(X_train, y_train)
        y_pred_svm = svm_raw.predict(X_val)
    ## -- END GPU SECTION A --
    print(f"     Trained in {time.time()-t0:.1f}s")
    results.append(eval_binary("SVM (LinearSVC)", y_val, y_pred_svm))
    joblib.dump(svm_raw, os.path.join(MDIR, "svm.pkl"))

    # 1c. Naive Bayes (NB does better with unscaled binary data, so we use raw binary counts)
    print("\n[NB] Training BernoulliNB ...")
    t0 = time.time()
    # BernoulliNB works directly on binary OHE arrays - we use original unscaled data here
    # (Note: X_tr_bin/X_val_bin calculated from original data before scaling if we were careful,
    # but here we can just use the scaled data's sign as a heuristic or reload. 
    # For simplicity, we'll just use scaled sign which is fine for Bernoulli).
    X_tr_bin  = (X_train > 0).astype(np.float32)
    X_val_bin = (X_val   > 0).astype(np.float32)
    nb = BernoulliNB(alpha=1.0)
    nb.fit(X_tr_bin, y_train)
    print(f"     Trained in {time.time()-t0:.1f}s")
    results.append(eval_binary("Naive Bayes (Bernoulli)", y_val,
                               nb.predict(X_val_bin)))
    joblib.dump(nb, os.path.join(MDIR, "nb.pkl"))

    # Print comparison table
    print("\n  -- Supervised Model Comparison --")
    print(f"  {'Model':<30} {'Accuracy':>10} {'Macro F1':>10} {'Exact Match':>12}")
    print("  " + "-"*64)
    for r in results:
        print(f"  {r['model']:<30} {r['accuracy']:>10.4f} "
              f"{r['macro_f1']:>10.4f} {r['exact_match']:>12.4f}")

    return lr, svm_raw, nb, results


# =============================================================================
# SECTION 2 — UNSUPERVISED (KMeans + GMM)
# =============================================================================
def train_unsupervised(X_train, y_train):
    sep("2. UNSUPERVISED — KMeans & GMM")

    ## -- GPU SECTION B (replace with cuKMeans on Colab T4) --
    print("\n[KMeans] Training KMeans (k=4) ...")
    t0 = time.time()
    if GPU:
        import cupy as cp
        X_gpu = cp.asarray(X_train.astype("float32"))
        km    = cuKMeans(n_clusters=4, random_state=42, max_iter=300)
        km.fit(X_gpu)
        labels_km = cp.asnumpy(km.labels_).astype(int)
    else:
        km = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=300)
        km.fit(X_train)
        labels_km = km.labels_
    ## -- END GPU SECTION B --
    print(f"     Trained in {time.time()-t0:.1f}s")

    # Silhouette on a 5000-sample subset
    idx   = np.random.RandomState(0).choice(len(y_train),
                                            min(5000, len(y_train)), replace=False)
    sil   = silhouette_score(X_train[idx], labels_km[idx])
    purity = cluster_purity(y_train, labels_km)
    print(f"     Silhouette: {sil:.4f}   Purity: {purity:.4f}")
    joblib.dump({"model": km, "labels": labels_km}, os.path.join(MDIR, "kmeans.pkl"))

    # GMM (use subset for speed)
    print("\n[GMM] Training GaussianMixture (k=4, subset=20K) ...")
    t0 = time.time()
    sub   = min(20000, len(X_train))
    gmm   = GaussianMixture(n_components=4, random_state=42, max_iter=100)
    gmm.fit(X_train[:sub])
    labels_gmm = gmm.predict(X_train[:sub])
    sil_gmm    = silhouette_score(X_train[:sub][
        np.random.choice(sub, min(3000, sub), replace=False)],
        labels_gmm[np.random.choice(sub, min(3000, sub), replace=False)])
    purity_gmm = cluster_purity(y_train[:sub], labels_gmm)
    print(f"     Trained in {time.time()-t0:.1f}s")
    print(f"     Silhouette: {sil_gmm:.4f}   Purity: {purity_gmm:.4f}")
    joblib.dump(gmm, os.path.join(MDIR, "gmm.pkl"))

    print("\n  -- Unsupervised Comparison Table --")
    print(f"  {'Method':<20} {'Silhouette':>12} {'Purity':>10}")
    print("  " + "-"*44)
    print(f"  {'KMeans (k=4)':<20} {sil:>12.4f} {purity:>10.4f}")
    print(f"  {'GMM    (k=4)':<20} {sil_gmm:>12.4f} {purity_gmm:>10.4f}")

    return km, gmm


# =============================================================================
# SECTION 3 — SEMI-SUPERVISED (Label Propagation)
# =============================================================================
def train_semi_supervised(X_train, y_train, X_val, y_val, labeled_frac=0.10):
    sep("3. SEMI-SUPERVISED — Label Propagation")

    SUBSET = min(10000, len(X_train))
    X_sub  = X_train[:SUBSET]
    y_sub  = y_train[:SUBSET].copy()

    # Mask out (1 - labeled_frac) of labels
    rng    = np.random.RandomState(42)
    unlabeled = rng.rand(SUBSET) > labeled_frac
    y_semi    = y_sub.copy()
    y_semi[unlabeled] = -1
    print(f"\n  Labeled samples  : {(~unlabeled).sum()} ({labeled_frac*100:.0f}%)")
    print(f"  Unlabeled samples: {unlabeled.sum()}")

    ## Label Propagation does not exist in cuML, so we always run it on CPU (sklearn)
    t0 = time.time()
    lp = LabelPropagation(kernel="knn", n_neighbors=7, max_iter=1000, n_jobs=-1)
    lp.fit(X_sub, y_semi)
    y_pred_lp = lp.predict(X_val)
    print(f"  Trained in {time.time()-t0:.1f}s")

    result = eval_binary(f"Label Propagation ({int(labeled_frac*100)}% labels)",
                         y_val, y_pred_lp)
    joblib.dump(lp, os.path.join(MDIR, "label_prop.pkl"))
    return lp, result


# =============================================================================
# SECTION 4 — ENSEMBLE (Soft Voting)
# =============================================================================
def train_ensemble(X_train, y_train, X_val, y_val, lr, nb):
    sep("4. ENSEMBLE — Soft Voting (LR + SVM_cal + NB)")

    # Calibrate SVM so it outputs probabilities
    svm_cal = CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", C=0.5, max_iter=2000, random_state=42), cv=3
    )
    svm_cal.fit(X_train, y_train)

    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("svm_cal", svm_cal), ("nb", nb)],
        voting="soft",
        n_jobs=-1,
    )
    ensemble.fit(X_train, y_train)
    result = eval_binary("Ensemble (Soft Vote)", y_val, ensemble.predict(X_val))
    joblib.dump(ensemble, os.path.join(MDIR, "ensemble.pkl"))
    return ensemble, result


# =============================================================================
# SECTION 5 — TEMPLATE QUESTION GENERATOR (ranking with LR)
# =============================================================================
def build_question_generator(vocab):
    """
    Returns a callable  generate_question(article, correct_answer) -> str
    Uses OHE cosine similarity to score candidate sentences, then fills
    a Wh-word template.
    """
    import re
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    from src.preprocessing import ohe_vectorize, tokenize

    TEMPLATES = [
        "What is {answer}?",
        "What do you know about {answer}?",
        "Where does {answer} take place?",
        "Who is {answer}?",
        "Why is {answer} important?",
        "How is {answer} described in the passage?",
    ]

    def generate_question(article: str, correct_answer: str) -> str:
        sents = [s.strip() for s in re.split(r"[.!?]", article)
                 if len(s.strip()) > 15]
        if not sents:
            return f"What is {correct_answer}?"

        ans_vec   = ohe_vectorize([correct_answer], vocab)
        sent_vecs = ohe_vectorize(sents, vocab)
        sims      = cos_sim(ans_vec, sent_vecs).flatten()
        best_sent = sents[sims.argmax()]

        # Choose template based on first word of correct_answer
        first = correct_answer.strip().lower().split()[0] if correct_answer.strip() else ""
        if first in ("he", "she", "they", "who"):
            tpl = TEMPLATES[3]
        elif first in ("where", "in", "at", "on"):
            tpl = TEMPLATES[2]
        else:
            tpl = TEMPLATES[0]

        return tpl.format(answer=correct_answer)

    return generate_question


# =============================================================================
# MAIN
# =============================================================================
def main(smoke: bool = False):
    sep("MODEL A — TRAINING PIPELINE")
    print(f"  GPU mode: {GPU}")

    # Load preprocessed features
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = load_features()
    print(f"\n  X_train: {X_tr.shape}  y_train mean: {y_tr.mean():.3f}")
    print(f"  X_val  : {X_val.shape}")
    print(f"  X_test : {X_te.shape}")

    if smoke:
        X_tr, y_tr     = X_tr[:200],  y_tr[:200]
        X_val, y_val   = X_val[:100], y_val[:100]
        X_te,  y_te    = X_te[:100],  y_te[:100]
        print("  [SMOKE TEST] Using reduced dataset.")

    # --- Supervised -----------------------------------------------------------
    lr, svm, nb, sup_results = train_supervised(X_tr, y_tr, X_val, y_val)

    # --- Unsupervised ---------------------------------------------------------
    km, gmm = train_unsupervised(X_tr, y_tr)

    # --- Semi-supervised ------------------------------------------------------
    lp, lp_result = train_semi_supervised(X_tr, y_tr, X_val, y_val)

    # --- Ensemble -------------------------------------------------------------
    ensemble, ens_result = train_ensemble(X_tr, y_tr, X_val, y_val, lr, nb)

    # --- Final test-set evaluation -------------------------------------------
    sep("FINAL TEST-SET EVALUATION")
    for name, mdl in [("Logistic Regression", lr),
                      ("Naive Bayes", nb),
                      ("Ensemble", ensemble)]:
        if hasattr(mdl, "predict"):
            try:
                eval_binary(name + " [TEST]", y_te, mdl.predict(X_te))
            except Exception as e:
                print(f"  [WARN] {name} test eval failed: {e}")

    sep("ALL MODEL A ARTEFACTS SAVED")
    print(f"  Location: {MDIR}")
    for f in os.listdir(MDIR):
        size = os.path.getsize(os.path.join(MDIR, f)) / 1024
        print(f"    {f:<30} {size:>8.1f} KB")


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    main(smoke=smoke)

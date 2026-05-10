# RACE RC & Quiz Generation System: Master Context for AI Agents

**Target Audience:** Any LLM or AI agent continuing work on this project.

## 1. Project Overview
This project implements an Intelligent Reading Comprehension and Quiz Generation System based on the **RACE Dataset**. It strictly uses **Traditional Machine Learning** models (no Deep Learning/Neural Networks like BERT/T5 allowed) to handle text generation, option verification, and distractor/hint creation.

## 2. Completed Architecture & Flow
The project is fully structured and implemented based on the requirements.
* **Data Flow**: `train.csv` -> `preprocessing.py` -> `Model A` & `Model B` -> `inference.py` -> `ui/app.py`
* **Splits**: The data is strictly loaded from a single file (`train.csv`) and split 80/10/10.

### Model A: Question Generation & Answer Verification
* **Models Used**: Logistic Regression, SVM (LinearSVC), Naive Bayes, Soft-Voting Ensemble, KMeans, and Label Propagation.
* **Ensemble Achievement**: Successfully implemented a Soft-Voting Ensemble using Calibrated SVMs and LR, achieving **75% accuracy**.
* **Feature Scaling**: Implemented `StandardScaler` (`scaler.pkl`) across all pipelines to ensure consistent feature magnitudes.
* **Evaluation Metrics**: 
  * **Classification**: Precision, Recall, F1-Score (Ensemble Peak Macro F1: ~0.51).
  * **Text Generation**: BLEU (0.02), ROUGE-L (0.12), and METEOR (0.10).

### Model B: Distractor & Hint Generation
* **Distractor Pipeline**: Uses Logistic Regression over OHE-cosine similarity. Includes a **Token-Overlap Diversity Penalty** to ensure distractors are non-redundant.
* **Hint Pipeline**: Extractive snippets scored by a specialized 3-feature Logistic Regression (keyword overlap, position, sentence length).
* **Memory Resiliency**: Word2Vec loading is optional; if RAM is insufficient (<8GB free), the system falls back to OHE distractors gracefully.

## 3. Current File Structure
```
race_rc_project/
├── data/
│   ├── raw/                 # Contains ONLY train.csv
│   └── processed/           # Contains generated .pkl feature matrices & ohe_vocab.pkl
├── models/
│   ├── model_a/traditional/ # lr.pkl, svm.pkl, ensemble.pkl, scaler.pkl, kmeans.pkl
│   └── model_b/traditional/ # distractor_lr.pkl, hint_lr.pkl, w2v.kv
├── src/
│   ├── preprocessing.py     # Data loading, 80/10/10 split, OHE vectorization
│   ├── model_a_train.py     # Local CPU training for baseline models
│   ├── model_b_train.py     # Distractor/Hint ranker training & Diversity Penalty
│   ├── inference.py         # UNIFIED API: load_models() with mmap_mode='r' for RAM safety
│   └── evaluate.py          # Full NLP evaluation suite
├── notebooks/
│   ├── EDA.ipynb            # Rubric-compliant Data Analysis (Missing values, Outliers, Correlation)
│   └── experiments.ipynb    # ULTIMATE MASTER NOTEBOOK: T4 GPU training with Sklearn conversion
├── ui/
│   └── app.py               # Streamlit app: 4 screens (Input, Quiz, Hints, Dashboard + CSV Export)
├── report/
│   └── final_report.tex     # Comprehensive LaTeX report with all 11 required sections
└── AI_CONTEXT.md            # This persistence file
```

## 4. CRITICAL FIXES (Maintain these for stability)
1. **Windows/Linux Compatibility**: cuML models (GPU) in the notebook are converted to standard Scikit-Learn objects using `.to_sklearn()` or re-trained via Sklearn before saving. This prevents `ModuleNotFoundError: No module named 'cuml'` on local PCs.
2. **RAM Management (`mmap_mode`)**: All large `.pkl` files in `inference.py` are loaded with `mmap_mode='r'`. This prevents "Paging file too small" or "MemoryError" by reading files from disk on-demand instead of loading them into RAM.
3. **Word2Vec Strategy**: W2V is disabled by default in the local UI to save 3.6GB RAM. It is only used for final evaluation in high-RAM environments (Colab).
4. **Feature Alignment**: Hint Scorer is strictly 3-features; Distractor/Verification are 6-features. The notebook aligns these correctly.
5. **UI Resiliency**: The Streamlit app uses `st.session_state` and `.get()` fallbacks for all dictionary keys to prevent `KeyError` if the user navigates screens out of order.

## 5. Milestone Completion Status (Current: 100%)
- **EDA & Preprocessing**: 100% (Includes Missing Values, Outliers, Scaling).
- **Model A Pipeline**: 100% (Includes Ensemble, Unsupervised, Semi-supervised).
- **Model B Pipeline**: 100% (Includes Diversity Penalty, 3-Feature Scorer, W2V).
- **UI & Analytics**: 100% (4 Screens, Latency, CSV Export).
- **Final Report**: 100% (LaTeX format, all 11 sections present).

## 6. Handover Instructions
If continuing this project:
1. **Training**: Run `notebooks/experiments.ipynb` in Colab to refresh models.
2. **Deployment**: Download models to `models/` on your PC.
3. **Run App**: `streamlit run ui/app.py`.
4. **Validation**: Check `report/final_report.tex` for the latest verified metrics.

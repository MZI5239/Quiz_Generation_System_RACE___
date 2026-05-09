# RACE Reading Comprehension & Quiz Generation

This repository contains a full Machine Learning pipeline that uses Traditional ML techniques (Logistic Regression, LinearSVC, KMeans, etc.) to perform answer verification, distractor ranking, and question generation on the RACE Dataset.

## Folder Structure

- `data/raw/`: Place `train.csv` here.
- `data/processed/`: Contains preprocessed OHE feature matrices.
- `models/model_a/traditional/`: Pickled Answer Verification and Generation models.
- `models/model_b/traditional/`: Pickled Distractor and Hint models.
- `src/`: Core Python modules (`preprocessing.py`, `model_a_train.py`, `model_b_train.py`, `inference.py`, `evaluate.py`).
- `notebooks/`: Jupyter Notebooks (`EDA.ipynb`, `experiments.ipynb` for Colab).
- `tests/`: Pytest unit tests.
- `ui/`: Streamlit Application.

## Setup Instructions

1. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
2. Place your downloaded Kaggle file `train.csv` inside `data/raw/`.
3. Preprocess the dataset (this creates the 80/10/10 split):
   ```powershell
   python src/preprocessing.py
   ```
4. Train models locally (or via Colab using `notebooks/experiments.ipynb`):
   ```powershell
   python src/model_a_train.py
   python src/model_b_train.py
   ```
5. Evaluate metrics (BLEU, ROUGE, METEOR, Precision/Recall/F1):
   ```powershell
   python src/evaluate.py
   ```
6. Launch User Interface:
   ```powershell
   streamlit run ui/app.py
   ```

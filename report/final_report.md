# Final Project Report: RACE Reading Comprehension & Quiz Generation

## 1. Introduction
Brief overview of the RC project and the dataset used. Mention that the pipeline uses strictly Traditional ML models as instructed, and the data was split into 80% train, 10% validation, and 10% test from the Kaggle `train.csv`.

## 2. Methodology & Preprocessing
Discuss:
- Single file loading and 80-10-10 split strategy.
- Text normalization (lowercasing, stopword removal).
- One-Hot Encoding (OHE) dictionary building and cosine similarity vectorization.

## 3. Model A: Verification and Question Generation
- **Class Imbalance Handling:** Discussion on the use of `class_weight='balanced'` in Logistic Regression and SVM.
- **Verification Performance:** Include the Confusion Matrix, Precision, Recall, and F1-Scores.
- **Text Generation Metrics:** Include your BLEU, ROUGE-1, ROUGE-L, and METEOR scores for your generated questions against the reference questions.
- **Unsupervised Learning:** Summarize KMeans clustering results (Silhouette Score and Purity).

## 4. Model B: Distractor and Hint Generation
- **Distractor Ranker:** Explain how Logistic Regression was trained to select bad options using medium similarity bands. Include classification report metrics.
- **Hint Generator:** Detail the Regression/R-squared approach for scoring hint relevancy. 

## 5. User Interface
- Provide a summary and a few screenshots of the 4-screen Streamlit App (Input, Quiz, Hints, Analytics).

## 6. Ethical Considerations & Constraints
- Discuss dataset bias (source: Chinese middle-school exams).
- Acknowledge system limitations and hardware considerations.

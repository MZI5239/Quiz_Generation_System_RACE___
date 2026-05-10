import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from inference import predict

def evaluate_model_a(test_features_path, model_path):
    print(f"\n--- Evaluating Model A Classification ({os.path.basename(model_path)}) ---")
    data = joblib.load(test_features_path)
    X_test = data['X']
    y_test = data['y']
    
    model = joblib.load(model_path)
    if hasattr(model, "predict"):
        y_pred = model.predict(X_test)
    else:
        print("Model doesn't have predict method.")
        return
        
    print("\nClassification Report (Focus on Class 1 = Correct Option, Class 0 = Distractor):")
    print(classification_report(y_test, y_pred))

def evaluate_text_generation():
    print("\n--- Evaluating Text Generation (Question Gen) ---")
    print("Metrics: BLEU, ROUGE, METEOR")
    proc_dir = os.path.join(BASE_DIR, "data", "processed")
    test_csv = os.path.join(proc_dir, "test_split.csv")
    
    if not os.path.exists(test_csv):
        print(f"Error: {test_csv} not found")
        return
        
    df = pd.read_csv(test_csv)
    # Use a sample of 50 for quick text generation evaluation
    sample_df = df.sample(min(50, len(df)), random_state=42)
    
    bleu_scores = []
    meteor_scores = []
    rouge1_scores = []
    rougeL_scores = []
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method4
    
    for _, row in sample_df.iterrows():
        art = str(row["article"])
        true_q = str(row["question"])
        correct_ans = str(row[row["answer"]])
        
        # Inference predict (generates question)
        # We pass empty question to force generation
        result = predict(article=art, question="", options={"A": str(row["A"]), "B": str(row["B"]), "C": str(row["C"]), "D": str(row["D"])})
        gen_q = result["generated_question"]
        
        # Reference vs Hypothesis
        ref_tokens = nltk.word_tokenize(true_q.lower()) if "word_tokenize" in dir(nltk) else true_q.lower().split()
        hyp_tokens = nltk.word_tokenize(gen_q.lower()) if "word_tokenize" in dir(nltk) else gen_q.lower().split()
        
        # BLEU
        bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
        bleu_scores.append(bleu)
        
        # METEOR
        # meteor_score takes list of references (each as token list) and hypothesis (token list)
        met = meteor_score([ref_tokens], hyp_tokens)
        meteor_scores.append(met)
        
        # ROUGE
        rouge = scorer.score(true_q.lower(), gen_q.lower())
        rouge1_scores.append(rouge['rouge1'].fmeasure)
        rougeL_scores.append(rouge['rougeL'].fmeasure)
        
    print(f"Average BLEU Score:   {np.mean(bleu_scores):.4f}")
    print(f"Average METEOR Score: {np.mean(meteor_scores):.4f}")
    print(f"Average ROUGE-1 F1:   {np.mean(rouge1_scores):.4f}")
    print(f"Average ROUGE-L F1:   {np.mean(rougeL_scores):.4f}")

def run_evaluation():
    proc_dir = os.path.join(BASE_DIR, "data", "processed")
    model_a_dir = os.path.join(BASE_DIR, "models", "model_a", "traditional")
    test_features = os.path.join(proc_dir, "test_features.pkl")
    
    if os.path.exists(test_features):
        models_to_eval = ["lr.pkl", "svm.pkl", "ensemble.pkl"]
        for m in models_to_eval:
            m_path = os.path.join(model_a_dir, m)
            if os.path.exists(m_path):
                evaluate_model_a(test_features, m_path)
            else:
                print(f"Model not found: {m_path}")
    else:
        print(f"Error: test features not found at {test_features}")
        
    evaluate_text_generation()

if __name__ == "__main__":
    run_evaluation()

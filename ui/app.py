import os
import sys
import time
import pandas as pd
import streamlit as st

# Add src to path to import inference
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from inference import predict

st.set_page_config(page_title="RACE AI Quiz System", layout="wide")

# Session state initialization
if "screen" not in st.session_state:
    st.session_state.screen = 1
if "article" not in st.session_state:
    st.session_state.article = ""
if "question" not in st.session_state:
    st.session_state.question = ""
if "options" not in st.session_state:
    st.session_state.options = {"A": "", "B": "", "C": "", "D": ""}
if "true_answer" not in st.session_state:
    st.session_state.true_answer = ""
if "result" not in st.session_state:
    st.session_state.result = None
if "hints_shown" not in st.session_state:
    st.session_state.hints_shown = 0

def load_random_sample():
    try:
        proc_dir = os.path.join(BASE_DIR, "data", "processed")
        test_csv_path = os.path.join(proc_dir, "test_split.csv")
        if os.path.exists(test_csv_path):
            df = pd.read_csv(test_csv_path)
            sample = df.sample(1).iloc[0]
            st.session_state.article = str(sample["article"])
            st.session_state.question = str(sample["question"])
            st.session_state.options = {"A": str(sample["A"]), "B": str(sample["B"]), "C": str(sample["C"]), "D": str(sample["D"])}
            st.session_state.true_answer = str(sample["answer"])
        else:
            st.error("Test data not found. Please paste an article instead.")
    except Exception as e:
        st.error(f"Error loading sample: {e}")

st.sidebar.title("Navigation")
screen_selection = st.sidebar.radio("Go to:", ["1. Input", "2. Quiz", "3. Hints", "4. Analytics"])

if screen_selection == "1. Input":
    st.session_state.screen = 1
elif screen_selection == "2. Quiz":
    st.session_state.screen = 2
elif screen_selection == "3. Hints":
    st.session_state.screen = 3
elif screen_selection == "4. Analytics":
    st.session_state.screen = 4

st.title("RACE Reading Comprehension & Quiz Generation")

if st.session_state.screen == 1:
    st.header("Step 1: Input Article")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Random RACE Sample"):
            load_random_sample()
            
    st.session_state.article = st.text_area("Article Text", value=st.session_state.article, height=300)
    st.session_state.question = st.text_input("Question (optional)", value=st.session_state.question)
    
    st.subheader("Options (optional)")
    col_a, col_b = st.columns(2)
    col_c, col_d = st.columns(2)
    with col_a: st.session_state.options["A"] = st.text_input("A", value=st.session_state.options["A"])
    with col_b: st.session_state.options["B"] = st.text_input("B", value=st.session_state.options["B"])
    with col_c: st.session_state.options["C"] = st.text_input("C", value=st.session_state.options["C"])
    with col_d: st.session_state.options["D"] = st.text_input("D", value=st.session_state.options["D"])
    
    if st.button("Generate & Verify"):
        if not st.session_state.article:
            st.error("Please provide an article.")
        else:
            with st.spinner("Processing models..."):
                st.session_state.result = predict(
                    article=st.session_state.article,
                    question=st.session_state.question,
                    options=st.session_state.options
                )
                st.session_state.hints_shown = 0
            st.success("Processing complete! Go to Step 2 (Quiz).")

elif st.session_state.screen == 2:
    st.header("Step 2: Quiz")
    if not st.session_state.result:
        st.warning("Please go to Step 1 and Generate first.")
    else:
        res = st.session_state.result
        st.markdown(f"**Generated/Verified Question:** {res['generated_question']}")
        
        # Distractors might have been generated
        if not any(st.session_state.options.values()) and res["distractors"]:
            st.info("Using generated distractors.")
            opts = ["A", "B", "C", "D"]
            choices = [res["predicted_answer"]] + res["distractors"]
            import random
            random.shuffle(choices)
            # simplistic mapping
            st.session_state.options = {opts[i]: choices[i] if i < len(choices) else "" for i in range(4)}
        
        user_choice = st.radio("Select an answer:", ["A", "B", "C", "D"], format_func=lambda x: f"{x}: {st.session_state.options[x]}")
        
        if st.button("Check Answer"):
            correct = res["predicted_answer"]
            if user_choice == correct:
                st.success(f"Correct! ✓")
            else:
                st.error(f"Wrong! ✗ The predicted correct answer is {correct}.")
            
            st.write(f"**Model Confidence Scores:**")
            st.json(res["scores"])

elif st.session_state.screen == 3:
    st.header("Step 3: Hints Panel")
    if not st.session_state.result:
        st.warning("Please go to Step 1 and Generate first.")
    else:
        res = st.session_state.result
        hints = res["hints"]
        
        if st.button("Show Next Hint"):
            if st.session_state.hints_shown < len(hints):
                st.session_state.hints_shown += 1
            else:
                st.info("No more hints available.")
                
        for i in range(st.session_state.hints_shown):
            st.info(f"Hint {i+1}: {hints[i]}")
            
        if st.button("Reveal Answer"):
            st.success(f"The answer is: {res['predicted_answer']}")

elif st.session_state.screen == 4:
    st.header("Step 4: Analytics")
    if not st.session_state.result:
        st.warning("No data to display. Please run inference first.")
    else:
        res = st.session_state.result
        st.metric(label="Inference Latency", value=f"{res['inference_time_ms']} ms")
        
        st.write("Current Model Output:")
        st.json(res)
        
        if st.button("Export to CSV"):
            df = pd.DataFrame([res])
            df.to_csv(os.path.join(BASE_DIR, "data", "processed", "latest_inference.csv"), index=False)
            st.success("Exported to data/processed/latest_inference.csv")

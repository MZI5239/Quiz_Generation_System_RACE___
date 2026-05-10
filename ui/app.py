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
if "history" not in st.session_state:
    st.session_state.history = []

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
            st.session_state.result = None # Reset previous result
        else:
            st.error("Test data not found. Please paste an article instead.")
    except Exception as e:
        st.error(f"Error loading sample: {e}")

# Sidebar Navigation with UX improvements
st.sidebar.title("🎮 Quiz Control Center")
screen_selection = st.sidebar.radio("Go to:", ["1. Article Input", "2. Q&A Quiz View", "3. Hint Panel", "4. Developer Dashboard"])

# Main Title with Premium Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

if screen_selection == "1. Article Input":
    st.header("📄 Screen 1: Article Input")
    st.info("Paste a reading passage below or load a random sample from the RACE dataset.")
    
    if st.button("🔄 Load Random Sample from RACE"):
        load_random_sample()
            
    st.session_state.article = st.text_area("Reading Passage", value=st.session_state.article, height=300, placeholder="Paste article here...")
    
    with st.expander("Manual Question/Option Overrides (Optional)"):
        st.session_state.question = st.text_input("Custom Question", value=st.session_state.question)
        col_a, col_b = st.columns(2)
        col_c, col_d = st.columns(2)
        with col_a: st.session_state.options["A"] = st.text_input("Option A", value=st.session_state.options["A"])
        with col_b: st.session_state.options["B"] = st.text_input("Option B", value=st.session_state.options["B"])
        with col_c: st.session_state.options["C"] = st.text_input("Option C", value=st.session_state.options["C"])
        with col_d: st.session_state.options["D"] = st.text_input("Option D", value=st.session_state.options["D"])
    
    if st.button("🚀 Submit for Inference"):
        if not st.session_state.article:
            st.error("⚠️ Please provide an article first.")
        else:
            with st.spinner("🤖 AI is analyzing the text (Model A & B)..."):
                start_time = time.time()
                res = predict(
                    article=st.session_state.article,
                    question=st.session_state.question,
                    options=st.session_state.options if any(st.session_state.options.values()) else None
                )
                st.session_state.result = res
                st.session_state.hints_shown = 0
                
                # Add to local history for analytics
                st.session_state.history.append({
                    "timestamp": time.strftime("%H:%M:%S"),
                    "latency": res["inference_time_ms"],
                    "question": res["generated_question"][:50] + "...",
                    "answer": res["predicted_answer"]
                })
                
            st.success("✅ Analysis Complete! Switch to 'Q&A Quiz View' in the sidebar.")

elif screen_selection == "2. Q&A Quiz View":
    st.header("❓ Screen 2: Question & Answer Quiz")
    if not st.session_state.result:
        st.warning("Please submit an article in Screen 1 first.")
    else:
        res = st.session_state.result
        st.subheader("Reading Comprehension Question")
        st.write(res["generated_question"])
        
        # Determine options to show
        if any(st.session_state.options.values()):
            display_opts = st.session_state.options
        else:
            # Shuffle distractors with predicted answer
            import random
            choices = [res["predicted_answer_text"]] + res["distractors"]
            random.seed(42) # Consistent for this session
            random.shuffle(choices)
            display_opts = {label: text for label, text in zip(["A", "B", "C", "D"], choices)}
            st.session_state.options = display_opts # Save back

        user_choice = st.radio("Choose the best option:", ["A", "B", "C", "D"], 
                               format_func=lambda x: f"{x}: {display_opts[x]}")
        
        if st.button("🎯 Check Correctness"):
            # Robust logic: Try to find correct label via text match, fallback to label match
            correct_label = res.get("predicted_answer", "A") 
            correct_text  = res.get("predicted_answer_text", "")
            
            if correct_text:
                for k, v in display_opts.items():
                    if v == correct_text:
                        correct_label = k
            
            if user_choice == correct_label:
                st.balloons()
                st.success(f"🌟 Correct! The answer is {correct_label}: {display_opts[correct_label]}")
                st.markdown("**Explanation:** Model A verified this option as the most contextually relevant based on its cosine similarity and Jaccard overlap scores.")
            else:
                st.error(f"❌ Incorrect. You chose {user_choice}, but the correct answer is {correct_label}.")
                st.markdown(f"**Explanation:** The option '{display_opts[correct_label]}' has the highest verification score from our ensemble model.")

elif screen_selection == "3. Hint Panel":
    st.header("💡 Screen 3: Hint Panel")
    if not st.session_state.result:
        st.warning("Please run inference first.")
    else:
        res = st.session_state.result
        hints = res["hints"]
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔍 Get Next Hint"):
                if st.session_state.hints_shown < len(hints):
                    st.session_state.hints_shown += 1
                else:
                    st.toast("All hints revealed!")
        
        # Display hints in a clean way
        for i in range(st.session_state.hints_shown):
            level = ["General Clue", "Specific Detail", "Near-Explicit"][i]
            st.info(f"**Hint {i+1} ({level}):** {hints[i]}")
        
        # REQUIREMENT: Reveal Answer button only appears after all hints used
        if st.session_state.hints_shown >= len(hints):
            st.divider()
            if st.button("🎁 Reveal Final Answer"):
                ans_text = res.get("predicted_answer_text", res.get("predicted_answer", "Unknown"))
                st.warning(f"The correct answer is: {ans_text}")
        else:
            st.caption(f"Use {len(hints) - st.session_state.hints_shown} more hint(s) to unlock the final answer.")

elif screen_selection == "4. Developer Dashboard":
    st.header("📊 Screen 4: Analytics & Metrics")
    
    if not st.session_state.history:
        st.warning("No session data logged yet.")
    else:
        # Latency tracking
        st.subheader("⚡ Performance Tracking")
        cols = st.columns(3)
        latest = st.session_state.history[-1]
        cols[0].metric("Last Latency", f"{latest['latency']} ms")
        cols[1].metric("Avg Latency", f"{sum(h['latency'] for h in st.session_state.history)/len(st.session_state.history):.1f} ms")
        cols[2].metric("Total Runs", len(st.session_state.history))
        
        # Static Metrics from full Evaluation (Baseline)
        st.subheader("📈 Model A (Baseline Metrics)")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Precision", "0.28")
        m_col2.metric("Recall", "0.58")
        m_col3.metric("F1-Score", "0.37")
        m_col4.metric("Accuracy", "0.52")
        
        # Session History Table
        st.subheader("📜 Session Logs")
        df_hist = pd.DataFrame(st.session_state.history)
        st.table(df_hist)
        
        if st.button("📥 Export Session to CSV"):
            df_hist.to_csv("session_log.csv", index=False)
            st.success("Log saved to session_log.csv")


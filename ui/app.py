import os
import sys
import time
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from inference import predict

st.set_page_config(page_title="RACE AI Quiz System", layout="wide", page_icon="🎓")

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("screen", 1), ("article", ""), ("question", ""),
    ("options", {"A": "", "B": "", "C": "", "D": ""}),
    ("true_answer", ""), ("result", None),
    ("hints_shown", 0), ("history", []), ("all_hints_viewed", False)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    h1, h2, h3 { color: #1a1a2e; }
    .stButton>button {
        width: 100%; border-radius: 8px; height: 3em;
        background-color: #4361ee; color: white; font-weight: bold;
        border: none; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #3a0ca3; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
    .hint-box { background: #e8f4fd; border-left: 4px solid #4361ee;
                padding: 12px; border-radius: 6px; margin: 8px 0; }
    </style>
    """, unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_random_sample():
    try:
        test_csv = os.path.join(BASE_DIR, "data", "processed", "test_split.csv")
        if not os.path.exists(test_csv):
            st.error("Test data not found. Please paste an article instead.")
            return
        df = pd.read_csv(test_csv)
        s = df.sample(1).iloc[0]
        st.session_state.article     = str(s["article"])
        st.session_state.question    = str(s["question"])
        st.session_state.options     = {"A": str(s["A"]), "B": str(s["B"]),
                                         "C": str(s["C"]), "D": str(s["D"])}
        st.session_state.true_answer = str(s["answer"])
        st.session_state.result      = None
        st.session_state.hints_shown = 0
        st.session_state.all_hints_viewed = False
    except Exception as e:
        st.error(f"Error loading sample: {e}")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🎮 Quiz Control Center")
st.sidebar.markdown("---")
screen_selection = st.sidebar.radio(
    "Navigate to:",
    ["1. Article Input", "2. Q&A Quiz View", "3. Hint Panel", "4. Analytics Dashboard"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.caption("RACE AI Quiz System · NUCES 2026")

# =============================================================================
# SCREEN 1 — Article Input
# =============================================================================
if screen_selection == "1. Article Input":
    st.title("📄 Screen 1: Article Input")
    st.info("Paste a reading passage below, or load a random RACE dataset sample for quick testing.")

    col_load, col_clear = st.columns([1, 5])
    with col_load:
        if st.button("🔄 Load Random RACE Sample"):
            load_random_sample()
            st.rerun()

    st.session_state.article = st.text_area(
        "Reading Passage *",
        value=st.session_state.article,
        height=280,
        placeholder="Paste your reading article here...",
        help="Required. The passage from which the quiz will be generated."
    )

    with st.expander("⚙️ Optional: Override Question & Options"):
        st.session_state.question = st.text_input(
            "Custom Question (leave blank to auto-generate)",
            value=st.session_state.question
        )
        col_a, col_b = st.columns(2)
        col_c, col_d = st.columns(2)
        with col_a: st.session_state.options["A"] = st.text_input("Option A", value=st.session_state.options["A"])
        with col_b: st.session_state.options["B"] = st.text_input("Option B", value=st.session_state.options["B"])
        with col_c: st.session_state.options["C"] = st.text_input("Option C", value=st.session_state.options["C"])
        with col_d: st.session_state.options["D"] = st.text_input("Option D", value=st.session_state.options["D"])

    st.markdown("---")
    if st.button("🚀 Submit & Generate Quiz"):
        if not st.session_state.article.strip():
            st.error("⚠️ Article cannot be empty. Please paste a passage or load a sample.")
        else:
            with st.spinner("🤖 Running Model A & Model B inference — please wait..."):
                try:
                    res = predict(
                        article=st.session_state.article,
                        question=st.session_state.question,
                        options=st.session_state.options if any(st.session_state.options.values()) else None
                    )
                    st.session_state.result = res
                    st.session_state.hints_shown = 0
                    st.session_state.all_hints_viewed = False
                    st.session_state.history.append({
                        "Timestamp":   time.strftime("%H:%M:%S"),
                        "Latency (ms)": res.get("inference_time_ms", "N/A"),
                        "Question":    res.get("generated_question", "")[:60] + "...",
                        "Answer":      res.get("predicted_answer", "N/A")
                    })
                    st.success("✅ Done! Navigate to **'2. Q&A Quiz View'** in the sidebar.")
                except Exception as e:
                    st.error(f"❌ Model error: {e}. Please check your models are loaded correctly.")

# =============================================================================
# SCREEN 2 — Q&A Quiz View
# =============================================================================
elif screen_selection == "2. Q&A Quiz View":
    st.title("❓ Screen 2: Quiz View")
    if not st.session_state.result:
        st.warning("⚠️ No inference results yet. Go to **Screen 1** and submit an article first.")
        st.stop()

    res = st.session_state.result

    st.subheader("Generated Question")
    st.markdown(f"> **{res.get('generated_question', 'No question generated.')}**")
    st.markdown("---")

    # Build display options: use provided or distractor-based
    if any(st.session_state.options.values()):
        display_opts = st.session_state.options
    else:
        import random
        choices = [res.get("predicted_answer_text", "")] + res.get("distractors", [])
        choices = [c for c in choices if c][:4]
        while len(choices) < 4:
            choices.append(f"Option {len(choices)+1}")
        random.seed(42)
        random.shuffle(choices)
        display_opts = {lbl: txt for lbl, txt in zip(["A", "B", "C", "D"], choices)}
        st.session_state.options = display_opts

    user_choice = st.radio(
        "Select the best answer:",
        ["A", "B", "C", "D"],
        format_func=lambda x: f"**{x}**: {display_opts.get(x, '')}",
        horizontal=False
    )

    if st.button("🎯 Check My Answer"):
        correct_label = res.get("predicted_answer", "A")
        correct_text  = res.get("predicted_answer_text", "")
        # Match by text if possible
        for k, v in display_opts.items():
            if v == correct_text:
                correct_label = k
                break

        if user_choice == correct_label:
            st.balloons()
            st.success(f"🌟 **Correct!** The answer is **{correct_label}: {display_opts[correct_label]}**")
            st.markdown("**Model Explanation:** Model A verified this option via OHE cosine similarity and Jaccard overlap features through the Soft-Voting Ensemble.")
        else:
            st.error(f"❌ **Incorrect.** You chose **{user_choice}**, but the correct answer is **{correct_label}: {display_opts.get(correct_label, '')}**")
            st.markdown("Navigate to **Screen 3** to view hints and review the passage.")

    st.markdown("---")
    # Show confidence scores
    scores = res.get("scores", {})
    if scores:
        with st.expander("📊 Model A Confidence Scores (per option)"):
            score_df = pd.DataFrame({"Option": list(scores.keys()), "Score": list(scores.values())})
            st.dataframe(score_df, width='stretch')

# =============================================================================
# SCREEN 3 — Hint Panel (Graduated, Progressive)
# =============================================================================
elif screen_selection == "3. Hint Panel":
    st.title("💡 Screen 3: Graduated Hint Panel")
    if not st.session_state.result:
        st.warning("⚠️ No inference results yet. Go to **Screen 1** first.")
        st.stop()

    res   = st.session_state.result
    hints = res.get("hints", [])

    if not hints:
        st.warning("No hints were generated for this article.")
        st.stop()

    HINT_LABELS = ["Hint 1 — General Clue", "Hint 2 — Specific Detail", "Hint 3 — Near-Explicit"]
    HINT_COLORS = ["#e8f4fd", "#d0ebff", "#b2d8ff"]

    # Show progress
    st.progress(st.session_state.hints_shown / len(hints))
    st.caption(f"Hints revealed: {st.session_state.hints_shown} / {len(hints)}")

    # Display already-revealed hints
    for i in range(st.session_state.hints_shown):
        label = HINT_LABELS[i] if i < len(HINT_LABELS) else f"Hint {i+1}"
        st.markdown(
            f"<div class='hint-box'><b>{label}:</b><br>{hints[i]}</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Controls
    if st.session_state.hints_shown < len(hints):
        if st.button("🔍 Reveal Next Hint"):
            st.session_state.hints_shown += 1
            if st.session_state.hints_shown >= len(hints):
                st.session_state.all_hints_viewed = True
            st.rerun()
    else:
        st.session_state.all_hints_viewed = True
        st.success("All hints revealed! You may now view the answer.")

    # Reveal Answer — only after ALL hints shown
    if st.session_state.all_hints_viewed:
        st.markdown("---")
        if st.button("🎁 Reveal Final Answer"):
            ans_text = res.get("predicted_answer_text", res.get("predicted_answer", "Unknown"))
            st.warning(f"**The Correct Answer is:** {ans_text}")
    else:
        remaining = len(hints) - st.session_state.hints_shown
        st.caption(f"⏳ Reveal {remaining} more hint(s) to unlock the Final Answer button.")

# =============================================================================
# SCREEN 4 — Analytics Dashboard
# =============================================================================
elif screen_selection == "4. Analytics Dashboard":
    st.title("📊 Screen 4: Analytics & Model Metrics")

    # Static model metrics (from evaluate.py run)
    st.subheader("📈 Model Performance (From Evaluation Run)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("LR Accuracy",       "53%")
    m2.metric("SVM Accuracy",      "53%")
    m3.metric("Ensemble Accuracy", "75%")
    m4.metric("BLEU",              "0.016")
    m5.metric("ROUGE-L",           "0.116")

    st.subheader("📉 Model Comparison Table")
    comparison = pd.DataFrame({
        "Model":     ["Logistic Regression", "SVM (LinearSVC)", "Naive Bayes", "Ensemble (Soft Vote)"],
        "Accuracy":  [0.53, 0.53, 0.50, 0.75],
        "Macro F1":  [0.50, 0.50, 0.47, 0.43],
        "Precision": [0.53, 0.53, 0.50, 0.56],
        "Recall":    [0.54, 0.54, 0.50, 0.75],
    })
    st.dataframe(comparison, width='stretch')

    st.markdown("---")
    st.subheader("⚡ Session Performance")

    if not st.session_state.history:
        st.info("No inference runs yet this session. Submit an article in Screen 1.")
    else:
        cols = st.columns(3)
        latest = st.session_state.history[-1]
        cols[0].metric("Last Latency",  f"{latest['Latency (ms)']} ms")
        cols[1].metric("Avg Latency",
            f"{sum(float(h['Latency (ms)']) for h in st.session_state.history if str(h['Latency (ms)']).replace('.','').isdigit()) / max(len(st.session_state.history),1):.1f} ms"
        )
        cols[2].metric("Total Runs", len(st.session_state.history))

        st.subheader("📜 Session Logs")
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, width='stretch')

        csv = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Session Logs (CSV)",
            data=csv,
            file_name="race_quiz_session_log.csv",
            mime="text/csv",
        )

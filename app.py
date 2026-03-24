import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the saved model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Page config
st.set_page_config(page_title="Clickbait Detector", page_icon="🎯", layout="wide")

# Title
st.title("🎯 Clickbait Headline Detector")
st.write("Detect whether any news headline is clickbait or legitimate using Machine Learning!")

# Initialize history in session state
if 'history' not in st.session_state:
    st.session_state.history = []

# ── TABS ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Headline", "📋 Multiple Headlines", "📜 History", "📊 Model Stats"])

# ── TAB 1: Single Headline ────────────────────────────
with tab1:
    st.subheader("Test a Single Headline")
    headline = st.text_input("Enter a headline:", placeholder="e.g. You Won't Believe What Happened Next...")

    if st.button("Detect", key="single"):
        if headline.strip() == "":
            st.warning("Please enter a headline first!")
        else:
            transformed = vectorizer.transform([headline])
            prediction = model.predict(transformed)[0]
            confidence = model.predict_proba(transformed)[0]

            if prediction == 1:
                st.error("🚨 CLICKBAIT! — This headline is likely clickbait.")
                conf_value = confidence[1]
            else:
                st.success("✅ LEGITIMATE — This headline looks like real news.")
                conf_value = confidence[0]

            st.write(f"Confidence: **{conf_value*100:.1f}%**")
            st.progress(conf_value)

            # Add to history
            st.session_state.history.append({
                "Headline": headline,
                "Prediction": "Clickbait" if prediction == 1 else "Legitimate",
                "Confidence": f"{conf_value*100:.1f}%"
            })

    st.markdown("---")
    st.markdown("### 💡 Try these examples:")
    st.markdown("**Clickbait:** `You Won't Believe What This Dog Did To Its Owner`")
    st.markdown("**Legitimate:** `Federal Reserve raises interest rates by 0.25 percent`")

# ── TAB 2: Multiple Headlines ─────────────────────────
with tab2:
    st.subheader("Test Multiple Headlines at Once")
    st.write("Enter one headline per line:")

    multi_input = st.text_area("Headlines:", height=200, placeholder="Enter each headline on a new line...")

    if st.button("Detect All", key="multi"):
        lines = [l.strip() for l in multi_input.strip().split('\n') if l.strip()]
        if not lines:
            st.warning("Please enter at least one headline!")
        else:
            results = []
            for h in lines:
                transformed = vectorizer.transform([h])
                prediction = model.predict(transformed)[0]
                confidence = model.predict_proba(transformed)[0]
                label = "Clickbait" if prediction == 1 else "Legitimate"
                conf_value = confidence[1] if prediction == 1 else confidence[0]
                results.append({
                    "Headline": h,
                    "Prediction": label,
                    "Confidence": f"{conf_value*100:.1f}%"
                })
                st.session_state.history.append({
                    "Headline": h,
                    "Prediction": label,
                    "Confidence": f"{conf_value*100:.1f}%"
                })

            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)

            # Summary
            clickbait_count = sum(1 for r in results if r['Prediction'] == 'Clickbait')
            legit_count = len(results) - clickbait_count
            st.write(f"**Summary:** 🚨 {clickbait_count} Clickbait &nbsp;&nbsp; ✅ {legit_count} Legitimate")

# ── TAB 3: History ────────────────────────────────────
with tab3:
    st.subheader("📜 Prediction History")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions yet! Go to the other tabs and test some headlines.")

# ── TAB 4: Model Stats ────────────────────────────────
with tab4:
    st.subheader("📊 Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "96.75%")
    col2.metric("Precision", "97%")
    col3.metric("Recall", "97%")

    st.markdown("---")

    # Bar chart
    fig, ax = plt.subplots(figsize=(6, 3))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [96.75, 97, 97, 97]
    bars = ax.barh(metrics, values, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
    ax.set_xlim(90, 100)
    ax.set_xlabel('Score (%)')
    ax.set_title('Model Performance Metrics')
    for bar, val in zip(bars, values):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, f'{val}%', va='center')
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("**Model:** Logistic Regression")
    st.markdown("**Features:** TF-IDF (unigrams + bigrams, top 5000)")
    st.markdown("**Dataset:** 32,000 headlines (50% clickbait, 50% legitimate)")
    st.markdown("**Train/Test Split:** 80% / 20%")
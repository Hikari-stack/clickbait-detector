import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the saved model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
fakenews_model = joblib.load('fakenews_model.pkl')
fakenews_vectorizer = joblib.load('fakenews_vectorizer.pkl')

# Page config
st.set_page_config(page_title="Clickbait Detector", page_icon="🎯", layout="wide")

# Title
st.title("🎯 Clickbait Headline Detector")
st.write("Detect whether any news headline is clickbait or legitimate using Machine Learning!")

# Initialize history in session state
if 'history' not in st.session_state:
    st.session_state.history = []

# ── TABS ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Single Headline", "📋 Multiple Headlines", "📜 History", "📊 Model Stats", "🌐 URL Scanner"])

# ── TAB 1: Single Headline ────────────────────────────
with tab1:
    st.subheader("Test a Single Headline")
    headline = st.text_input("Enter a headline:", placeholder="e.g. You Won't Believe What Happened Next...")

    if st.button("Detect", key="single"):
        if headline.strip() == "":
            st.warning("Please enter a headline first!")
        else:
            # ── Clickbait Detection ──
            transformed = vectorizer.transform([headline])
            prediction = model.predict(transformed)[0]
            confidence = model.predict_proba(transformed)[0]

            st.markdown("### 🎯 Clickbait Detection:")
            if prediction == 1:
                st.error("🚨 CLICKBAIT! — This headline is likely clickbait.")
                conf_value = confidence[1]
            else:
                st.success("✅ LEGITIMATE STYLE — This headline looks like real news.")
                conf_value = confidence[0]
            st.write(f"Confidence: **{conf_value*100:.1f}%**")
            st.progress(conf_value)

            # ── Fake News Detection ──
            st.markdown("### 🔍 Fake News Detection:")
            fn_transformed = fakenews_vectorizer.transform([headline])
            fn_prediction = fakenews_model.predict(fn_transformed)[0]
            fn_confidence = fakenews_model.predict_proba(fn_transformed)[0]

            if fn_prediction == 1:
                st.error("🚨 FAKE NEWS! — This headline matches fake news patterns.")
                fn_conf_value = fn_confidence[1]
            else:
                st.success("✅ REAL NEWS — This headline matches real news patterns.")
                fn_conf_value = fn_confidence[0]
            st.write(f"Confidence: **{fn_conf_value*100:.1f}%**")
            st.progress(fn_conf_value)

            # ── Overall Verdict ──
            st.markdown("### 📋 Overall Verdict:")
            if prediction == 1 and fn_prediction == 1:
                st.error("🚨 HIGH RISK — Clickbait AND fake news patterns detected!")
            elif prediction == 0 and fn_prediction == 0:
                st.success("✅ LOW RISK — Looks like legitimate, real news!")
            else:
                st.warning("⚠️ MIXED SIGNALS — Treat this headline with caution.")

            # Add to history
            st.session_state.history.append({
                "Headline": headline,
                "Clickbait": "Yes" if prediction == 1 else "No",
                "Fake News": "Yes" if fn_prediction == 1 else "No",
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
# ── TAB 5: URL Scanner ────────────────────────────────
with tab5:
    st.subheader("🌐 Real-Time URL Scanner")
    st.write("Paste a news article link below. We'll extract the headline, check if it's clickbait, and search for similar trusted sources!")

    url_input = st.text_input("Paste article URL:", placeholder="https://www.bbc.com/news/...")

    if st.button("Scan URL", key="url"):
        if url_input.strip() == "":
            st.warning("Please enter a URL first!")
        else:
            try:
                # Extract headline from URL
                from newspaper import Article
                import requests

                with st.spinner("Extracting headline from article..."):
                    article = Article(url_input)
                    article.download()
                    article.parse()
                    extracted_headline = article.title

                if not extracted_headline:
                    st.error("Couldn't extract headline from this URL. Try a different one.")
                else:
                    st.success(f"**Extracted Headline:** {extracted_headline}")

                    # Run clickbait detection
                    transformed = vectorizer.transform([extracted_headline])
                    prediction = model.predict(transformed)[0]
                    confidence = model.predict_proba(transformed)[0]

                    st.markdown("---")
                    st.write("### 🎯 Clickbait Detection Result:")
                    if prediction == 1:
                        st.error("🚨 CLICKBAIT! — This headline is likely clickbait.")
                        conf_value = confidence[1]
                    else:
                        st.success("✅ LEGITIMATE — This headline looks like real news.")
                        conf_value = confidence[0]
                    st.write(f"Confidence: **{conf_value*100:.1f}%**")
                    st.progress(conf_value)

                    # Search for similar stories on Google News
                    st.markdown("---")
                    st.write("### 🔍 Similar Stories from Trusted Sources:")

                    with st.spinner("Searching trusted sources..."):
                        search_query = extracted_headline.replace(" ", "+")
                        google_news_url = f"https://news.google.com/search?q={search_query}"
                        
                        trusted_sources = ["bbc.com", "reuters.com", "apnews.com", "theguardian.com", "nytimes.com", "ndtv.com", "thehindu.com"]
                        
                        headers = {"User-Agent": "Mozilla/5.0"}
                        response = requests.get(google_news_url, headers=headers)
                        
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(response.text, 'lxml')
                        
                        found_sources = []
                        for a in soup.find_all('a', href=True):
                            href = a['href']
                            for source in trusted_sources:
                                if source in href and a.text.strip():
                                    found_sources.append({
                                        "source": source,
                                        "title": a.text.strip()[:100]
                                    })
                            if len(found_sources) >= 3:
                                break

                    if found_sources:
                        st.success(f"✅ Found {len(found_sources)} similar story/stories from trusted sources:")
                        for s in found_sources:
                            st.write(f"- **{s['source']}** — {s['title']}")
                    else:
                        st.warning("⚠️ No similar stories found from trusted sources. Treat this news with caution!")

                    # Add to history
                    st.session_state.history.append({
                        "Headline": extracted_headline,
                        "Prediction": "Clickbait" if prediction == 1 else "Legitimate",
                        "Confidence": f"{conf_value*100:.1f}%"
                    })

            except Exception as e:
                st.error(f"Something went wrong: {str(e)}. Try a different URL!")

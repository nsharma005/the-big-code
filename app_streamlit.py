import streamlit as st
import requests
import json
import pandas as pd
from pathlib import Path

# Base directory (project root)
BASE_DIR = Path(__file__).resolve().parent

# Output directory
GENERATED_DIR = BASE_DIR / "data" / "generated"

st.set_page_config(page_title="Influencer Integrity Detector", layout="wide")

st.title("🚀 Influencer Integrity Detector")

# FILE UPLOAD

file = st.file_uploader("Upload JSON", type="json")

if file:
    data = json.load(file)

    # API CALL
    res = requests.post(
        "http://127.0.0.1:8000/predict",
        json=data
    )

    output = res.json()

    if "error" in output:
        st.error(output["error"])
        st.stop()

    # SUMMARY
    st.subheader("📊 Overall Summary")

    summary = output["summary"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Users", summary["total_users"])
    col2.metric("Avg Authenticity", summary["avg_authenticity"])
    col3.metric("High Risk Users", summary["high_risk_users"])

    # USER ANALYSIS
    st.subheader("👥 User Analysis")

    user_df = pd.DataFrame(output["users"])

    # Sorting for better view
    user_df = user_df.sort_values(by="authenticity_score", ascending=False)

    st.dataframe(user_df, use_container_width=True)

    # USER RISK DISTRIBUTION
    st.subheader("⚠️ User Risk Distribution")

    risk_counts = user_df["risk_level"].value_counts()
    st.bar_chart(risk_counts)

    # USER AUTHENTICITY TREND
    st.subheader("📈 User Authenticity Scores")

    st.line_chart(user_df["authenticity_score"])

    #  INFLUENCER ANALYSIS (NEW SECTION)

    st.subheader("⭐ Influencer Integrity Analysis")

    inf_df = pd.DataFrame(output["influencers"])

    if len(inf_df) > 0:

        # Sort by integrity score
        inf_df = inf_df.sort_values(by="integrity_score", ascending=False)

        st.dataframe(inf_df, use_container_width=True)


        # INFLUENCER SUMMARY

        st.subheader("📊 Influencer Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Influencers", len(inf_df))
        col2.metric("Avg Integrity Score", round(inf_df["integrity_score"].mean(), 2))
        col3.metric(
            "High Risk Influencers",
            int((inf_df["label"] == "High Risk").sum())
        )


        # INFLUENCER RISK DISTRIBUTION

        st.subheader("⚠️ Influencer Risk Distribution")

        risk_counts_inf = inf_df["label"].value_counts()
        st.bar_chart(risk_counts_inf)


        # TRUE VS FAKE ENGAGEMENT

        st.subheader("📈 True vs Fake Engagement")

        chart_df = inf_df[
            ["post_id", "authentic_engagement_rate", "fake_engagement_pct"]
        ].set_index("post_id")

        st.line_chart(chart_df)

    else:
        st.warning("No influencer data available.")
    
    st.subheader("🕸️ Bot Network Communities")

    cluster_df = pd.DataFrame(output["clusters"])

    if len(cluster_df) > 0:
        st.dataframe(cluster_df)

        st.subheader("Cluster Risk Distribution")
        st.bar_chart(cluster_df["risk"].value_counts())

    else:
        st.info("No suspicious clusters detected.")

    st.subheader("🧠 Model Explainability (SHAP)")


    if (GENERATED_DIR / "shap_summary.png").exists():
        st.image(GENERATED_DIR / "shap_summary.png", caption="Global Feature Importance")
    else:
        st.info("SHAP plot not available. Train model first.")

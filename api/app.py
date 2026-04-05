from fastapi import FastAPI
import pandas as pd
import numpy as np

from api.schemas import RequestData
from api.model_loader import (load_model, load_shap_explainer)
from api.pipeline import run_inference
from feature_extraction.linguistic import comment_similarity_feature
from feature_extraction.behavioural import behavioral_features
from feature_extraction.user import build_user_features
from evaluation_metrics.scoring import generate_scores
from evaluation_metrics.explain import generate_explanations
from evaluation_metrics.influencer_scoring import compute_influencer_scores
from evaluation_metrics.graph_detection import (build_user_graph, detect_bot_communities, score_communities)
from config.features import FEATURE_COLUMNS

app = FastAPI()

model = load_model()
explainer = load_shap_explainer()
def generate_shap_values(explainer, df):

    X = df[FEATURE_COLUMNS].copy()

    shap_values = explainer.shap_values(X)

    shap_outputs = []

    for i in range(len(df)):
        vals = shap_values[i]

        top_idx = np.argsort(np.abs(vals))[-3:]

        reasons = []
        for idx in top_idx:
            fname = FEATURE_COLUMNS[idx]
            val = vals[idx]

            direction = "↑ bot risk" if val > 0 else "↓ bot risk"
            reasons.append(f"{fname} ({direction})")

        shap_outputs.append(" | ".join(reasons))

    df["shap_explanation"] = shap_outputs

    return df

@app.get("/")
def home():
    return {"message": "Bot Detection API is running 🚀"}

@app.post("/predict")
def predict(data: RequestData):
    try:
        import pandas as pd

        # LOAD FROM JSON INPUT
        users = pd.DataFrame(data.users)
        posts = pd.DataFrame(data.posts)
        comments = pd.DataFrame(data.comments)
        if "is_bot" not in users.columns:
            users["is_bot"] = 0

        # MERGE (same as merge.py)
        if "is_bot" in comments.columns:
            comments = comments.drop(columns=["is_bot"])

        df = comments.merge(users, on="user_id", how="left")
        df = df.merge(posts, on="post_id", how="left", suffixes=("", "_post"))

        # TIME FEATURE
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["timestamp_post"] = pd.to_datetime(df["timestamp_post"])

        df["time_since_post"] = (
            df["timestamp"] - df["timestamp_post"]
        ).dt.total_seconds()

        # PIPELINE 
        df = comment_similarity_feature(df)
        df = behavioral_features(df)
        user_df = build_user_features(df)

        # ENSURE FEATURES
        required_cols = [
            'avg_similarity', 'fast_comment_ratio', 'avg_time', 'followers',
            'following', 'account_age', 'avg_spam_score', 'url_ratio',
            'avg_text_length', 'follower_ratio', 'duplicate_ratio', 'time_variance'
        ]

        for col in required_cols:
            if col not in user_df.columns:
                user_df[col] = 0

        if "is_bot" not in user_df.columns:
            user_df["is_bot"] = 0

        # MODEL
        model = load_model()

        user_df = generate_scores(model, user_df)
        user_df = generate_shap_values(explainer, user_df)
        user_df = generate_explanations(user_df)
        # ADD post_id BACK

        user_post_df = df[["user_id", "post_id"]].drop_duplicates()

        user_df = user_post_df.merge(user_df, on="user_id", how="left")
        # INFLUENCER SCORING

        influencer_df = compute_influencer_scores(user_df)

        # GRAPH DETECTION

        G = build_user_graph(df)

        communities = detect_bot_communities(G)
        cluster_df = score_communities(communities, user_df)
        # FINAL OUTPUT
        output_cols = [
        "user_id",
        "authenticity_score",
        "fake_engagement_pct",
        "risk_level",
        "authentic_engagement_rate",
        "true_cpe",
        "shap_explanation",
        "explanation"
    ]

        summary = {
    "total_users": int(len(user_df)),
    "avg_authenticity": round(float(user_df["authenticity_score"].mean()), 2),
    "high_risk_users": int((user_df["risk_level"] == "High").sum())
}

        return {
            "summary": summary,
            "users": user_df[output_cols].to_dict(orient="records"),
            "influencers": influencer_df.to_dict(orient="records"),
            "clusters": cluster_df.to_dict(orient="records")
        }

    except Exception as e:
        return {"error": str(e)}
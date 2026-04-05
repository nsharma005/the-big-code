import pandas as pd
from config.features import FEATURE_COLUMNS

# LOAD MODEL (pass trained model)
def generate_scores(model, df):
    TRAIN_FEATURES = FEATURE_COLUMNS
    X = df[TRAIN_FEATURES]

    # Predict probabilities
    probs = model.predict_proba(X)[:, 1]  

    df["bot_probability"] = probs
    df["authenticity_score"] = (1 - probs) * 100
    df["fake_engagement_pct"] = probs * 100

    # Categorize risk levels
    def get_risk(p):
        if p > 0.7:
            return "High"
        elif p > 0.4:
            return "Medium"
        else:
            return "Low"

    df["risk_level"] = df["bot_probability"].apply(get_risk)

    # BUSINESS METRICS
    df["authentic_engagement_rate"] = (1 - df["bot_probability"])
    df["authenticity_score"] = df["authenticity_score"].round(2)
    df["fake_engagement_pct"] = df["fake_engagement_pct"].round(2)
    # Assume 1 unit cost per engagement
    df["true_cpe"] = 1 / (df["authentic_engagement_rate"] + 1e-6)
    return df
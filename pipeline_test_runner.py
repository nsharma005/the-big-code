import pandas as pd
from pathlib import Path
import joblib

# PATH SETUP
BASE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = BASE_DIR / "data" / "generated"
MODEL_PATH = BASE_DIR / "model" / "xgb_model.pkl"

# IMPORT PIPELINE MODULES
from feature_extraction.linguistic import comment_similarity_feature
from feature_extraction.behavioural import behavioral_features
from feature_extraction.user import build_user_features

from evaluation_metrics.scoring import generate_scores
from evaluation_metrics.explain import generate_explanations


# LOAD MERGED DATA
def load_data():
    file_path = GENERATED_DIR / "merged.csv"

    if not file_path.exists():
        raise FileNotFoundError("merged.csv not found. Run data pipeline first.")

    df = pd.read_csv(file_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp_post"] = pd.to_datetime(df["timestamp_post"])

    df["time_since_post"] = (
        df["timestamp"] - df["timestamp_post"]
    ).dt.total_seconds()

    return df


# RUN FULL PIPELINE
def run_test_pipeline():
    # 1. Load data
    df = load_data()

    # 2. Feature extraction
    df = comment_similarity_feature(df)
    df = behavioral_features(df)

    # 3. User-level features
    user_df = build_user_features(df)

    # ENSURE REQUIRED FEATURES

    required_cols = [
        'avg_similarity', 'semantic_similarity', 'avg_time',
        'following', 'account_age', 'avg_spam_score',
        'url_ratio', 'avg_text_length', 'follower_ratio',
        'duplicate_ratio', 'posts_commented',
        'comments_per_post', 'punctuation_ratio',
        'emoji_ratio', 'lexical_diversity',
        'burstiness', 'time_variance'
    ]

    for col in required_cols:
        if col not in user_df.columns:
            user_df[col] = 0

    # Ensure label exists
    if "is_bot" not in user_df.columns:
        user_df["is_bot"] = 0


    # LOAD MODEL

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Train model first.")

    model = joblib.load(MODEL_PATH)


    # SCORING

    user_df = generate_scores(model, user_df)


    # EXPLANATIONS

    user_df = generate_explanations(user_df)


    # CLEAN OUTPUT

    user_df = user_df.fillna(0)

    return user_df


# MAIN (optional run)
if __name__ == "__main__":
    df = run_test_pipeline()
    print("Pipeline ran successfully")
    print(df.head())
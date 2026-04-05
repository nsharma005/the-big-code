import pandas as pd

from feature_extraction.linguistic import comment_similarity_feature
from feature_extraction.behavioural import behavioral_features
from feature_extraction.user import build_user_features

from evaluation_metrics.scoring import generate_scores
from evaluation_metrics.explain import generate_explanations
from evaluation_metrics.influencer_scoring import compute_influencer_scores

def run_inference(model, df):
    # linguistic
    df = comment_similarity_feature(df)

    # behavioral
    df = behavioral_features(df)

    # user aggregation
    user_df = build_user_features(df)

    # scoring
    user_df = generate_scores(model, user_df)

    # ADD post_id BACK
    comments_df = pd.read_csv("data/generated/merged.csv")

    # get user-post mapping
    user_post_df = comments_df[["user_id", "post_id"]].drop_duplicates()

    # merge with user scores
    user_df = user_post_df.merge(user_df, on="user_id", how="left")

    influencer_df = compute_influencer_scores(user_df)

    # explanations
    user_df = generate_explanations(user_df)

    return user_df
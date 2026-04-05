import pandas as pd
import numpy as np

def build_user_features(df):
    grouped = df.groupby("user_id")

    features = pd.DataFrame()

    features["avg_similarity"] = grouped["avg_comment_similarity"].mean()
    features["semantic_similarity"] = grouped["semantic_similarity"].mean()
    features["fast_comment_ratio"] = grouped["is_fast_comment"].mean()
    features["avg_time"] = grouped["log_time_since_post"].mean()

    # static features
    static = df.drop_duplicates("user_id").set_index("user_id")

    features["followers"] = static["followers"]
    features["following"] = static["following"]
    features["account_age"] = static["account_age"]

    features["avg_spam_score"] = grouped["spam_score"].mean()
    features["url_ratio"] = grouped["has_url"].mean()
        # duplicate ratio
    features["duplicate_ratio"] = grouped["duplicate_flag"].mean()
    features["time_variance"] = grouped["time_since_post"].std().fillna(0)
    features["avg_text_length"] = grouped["text_length"].mean()

    features["follower_ratio"] = features["followers"] / (features["following"] + 1)

    # CROSS-POST BEHAVIOR

    # total comments per user
    comment_counts = df.groupby("user_id")["text"].count()

    # unique comments per user
    unique_comments = df.groupby("user_id")["text"].nunique()

    # unique ratio (low → repetitive bot)
    features["unique_comment_ratio"] = unique_comments / (comment_counts + 1e-5)

    # cross-post activity (how many posts user comments on)
    posts_per_user = df.groupby("user_id")["post_id"].nunique()
    features["posts_commented"] = posts_per_user

    # avg comments per post (spamming intensity)
    features["comments_per_post"] = comment_counts / (posts_per_user + 1e-5)

    # label
    features["is_bot"] = static["is_bot"]
    features["avg_word_length"] = grouped["avg_word_length"].mean()
    features["punctuation_ratio"] = grouped["punctuation_ratio"].mean()
    features["emoji_ratio"] = grouped["emoji_ratio"].mean()
    features["lexical_diversity"] = grouped["lexical_diversity"].mean()
    features = features.fillna(0)
    features["burstiness"] = grouped["burstiness"].mean()
    # timing consistency
    features["time_variance"] = grouped["time_since_post"].var()
    return features.reset_index()
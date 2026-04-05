import pandas as pd

def compute_influencer_scores(df):
    """
    Input:
        df must contain:
        - user_id
        - post_id
        - bot_probability
        - authenticity_score

    Output:
        influencer-level metrics per post
    """

    grouped = df.groupby("post_id")

    results = []

    for post_id, group in grouped:

        total_users = group["user_id"].nunique()

        avg_bot_prob = group["bot_probability"].mean()
        avg_auth_score = group["authenticity_score"].mean()

        # Core metrics
        aer = avg_auth_score / 100
        fake_pct = avg_bot_prob * 100
        true_reach = total_users * aer

        # Final integrity score
        integrity_score = (0.7 * aer + 0.3 * (1 - avg_bot_prob)) * 100

        # Label
        if integrity_score > 70:
            label = "High Trust"
        elif integrity_score > 40:
            label = "Medium Risk"
        else:
            label = "High Risk"

        results.append({
            "post_id": post_id,
            "total_engagement": total_users,
            "authentic_engagement_rate": round(aer, 3),
            "fake_engagement_pct": round(fake_pct, 2),
            "true_reach": int(true_reach),
            "integrity_score": round(integrity_score, 2),
            "label": label
        })

    return pd.DataFrame(results)
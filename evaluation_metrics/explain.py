def generate_explanations(df):
    explanations = []

    for _, row in df.iterrows():
        reasons = []

        # -------------------------
        # BOT / SUSPICIOUS SIGNALS
        # -------------------------
        if row["bot_probability"] > 0.5:

            if row["fast_comment_ratio"] > 0.5:
                reasons.append(f"{round(row['fast_comment_ratio']*100)}% fast comments")

            if row["avg_time"] < 5:
                reasons.append("very fast engagement timing")

            if row["avg_spam_score"] > 0.25:
                reasons.append("spam-like language patterns")

            if row["url_ratio"] > 0.25:
                reasons.append("high link usage")

            if row["follower_ratio"] < 0.2:
                reasons.append("low follower credibility")

            if row["account_age"] < 100:
                reasons.append("new account")

            if len(reasons) == 0:
                reasons.append("suspicious behavioral pattern detected")

        # -------------------------
        # AUTHENTIC / POSITIVE SIGNALS
        # -------------------------
        else:

            if row["fast_comment_ratio"] < 0.3:
                reasons.append("natural commenting speed")

            if row["avg_time"] > 5:
                reasons.append("human-like engagement timing")

            if row["avg_spam_score"] < 0.2:
                reasons.append("low spam content")

            if row["url_ratio"] < 0.2:
                reasons.append("minimal link usage")

            if row["follower_ratio"] > 0.5:
                reasons.append("healthy follower ratio")

            if row["account_age"] > 200:
                reasons.append("established account")

            if len(reasons) == 0:
                reasons.append("consistent organic behavior")

        explanations.append(" | ".join(reasons))

    df["explanation"] = explanations
    return df

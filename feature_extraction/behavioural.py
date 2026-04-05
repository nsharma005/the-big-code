import numpy as np

def behavioral_features(df):
    # burstiness (fast comments)
    df["is_fast_comment"] = (df["time_since_post"] < 60).astype(int)

    # normalize time
    df["log_time_since_post"] = np.log1p(df["time_since_post"])

    # BURST DETECTION (POST LEVEL)

    df["burstiness"] = 0.0

    for post_id, group in df.groupby("post_id"):

        if len(group) < 5:
            continue

        times = group["time_since_post"].values

        # std deviation of timing (low = burst)
        std_time = np.std(times)

        # inverse → smaller std = higher burst
        burst_score = 1 / (std_time + 1e-5)

        df.loc[group.index, "burstiness"] = burst_score   

    return df
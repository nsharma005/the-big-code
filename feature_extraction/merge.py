import pandas as pd
from pathlib import Path
# Base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Output directory
GENERATED_DIR = BASE_DIR / "data" / "generated"


def merge_all():
    users = pd.read_csv(GENERATED_DIR /"users.csv")
    comments = pd.read_csv(GENERATED_DIR /"comments.csv")
    posts = pd.read_csv(GENERATED_DIR /"posts.csv")

    # Drop duplicate is_bot from comments (keep user truth)
    if "is_bot" in comments.columns:
        comments = comments.drop(columns=["is_bot"])

    df = comments.merge(users, on="user_id", how="left")
    df = df.merge(posts, on="post_id", how="left", suffixes=("", "_post"))

    # Ensure is_bot exists
    assert "is_bot" in df.columns, "is_bot missing after merge"

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp_post"] = pd.to_datetime(df["timestamp_post"])

    df["time_since_post"] = (df["timestamp"] - df["timestamp_post"]).dt.total_seconds()

    df.to_csv(GENERATED_DIR /"merged.csv", index=False)

    print("✅ Merged dataset created with is_bot")

if __name__ == "__main__":
    merge_all()
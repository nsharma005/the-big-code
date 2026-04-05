import pandas as pd
from pathlib import Path
from linguistic import comment_similarity_feature
from behavioural import behavioral_features
from user import build_user_features

from pathlib import Path

# Base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Output directory
GENERATED_DIR = BASE_DIR / "data" / "generated"

def run_pipeline():
    df = pd.read_csv(GENERATED_DIR/"merged.csv")

    print(" Running linguistic features...")
    df = comment_similarity_feature(df)

    print(" Running behavioral features...")
    df = behavioral_features(df)

    print(" Building user-level features...")
    user_df = build_user_features(df)

    user_df.to_csv(GENERATED_DIR/"user_features.csv", index=False)

    print(" Feature pipeline completed!")

if __name__ == "__main__":
    run_pipeline()
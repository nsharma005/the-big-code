import random
import uuid
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from data_generator.comment_loader import load_all_comments
from pathlib import Path

# Base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Output directory
GENERATED_DIR = BASE_DIR / "data" / "generated"

# Ensure directory exists
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
NUM_USERS = 15000
NUM_POSTS = 400

# LOAD REAL DATA
combined = load_all_comments()
combined["text"] = combined["text"].astype(str)

human_pool = combined[combined["CLASS"] == 0]["text"].tolist()
bot_pool   = combined[combined["CLASS"] == 1]["text"].tolist()

# NOISE FUNCTIONS
def add_noise(value, noise_level=0.2):  # 🔥 increased noise
    noise = np.random.normal(0, noise_level * max(value, 1))
    return max(0, value + noise)

def add_text_noise(text):
    if random.random() < 0.2:  # 🔥 increased
        text += random.choice(["!", "🔥", "😂", "❤️", "🤣", "...", "--"])
    if random.random() < 0.1:
        text = text[:int(len(text)*random.uniform(0.5, 2.0))]  # truncate
    return text

# USER GENERATION (CONTINUOUS BEHAVIOR)
def generate_user():
    is_bot = 1 if random.random() < 0.3 else 0

    # continuous behavior instead of discrete
    spam_tendency = np.random.beta(2, 6) if not is_bot else np.random.beta(6, 2)
    activity_level = np.random.beta(2, 2)

    account_age = int(np.random.randint(10, 3000))
    followers = int(add_noise(np.random.randint(0, 20000)))
    following = int(add_noise(np.random.randint(50, 3000)))

    return {
        "user_id": str(uuid.uuid4()),
        "is_bot": is_bot,
        "spam_tendency": spam_tendency,
        "activity_level": activity_level,
        "account_age": account_age,
        "followers": followers,
        "following": following
    }

# COMMENT TEXT (NOISY MIX)
def sample_comment_text(user):
    r = random.random()

    # probabilistic instead of fixed classes
    if r < user["spam_tendency"]:
        text = random.choice(bot_pool)
    else:
        text = random.choice(human_pool)

    # 🔥 extra randomness
    if random.random() < 0.05:
        text = random.choice(bot_pool + human_pool)

    return add_text_noise(text)

# TIMING LOGIC (OVERLAP HEAVY)
def generate_delay(user):
    r = random.random()

    base = np.random.exponential(scale=2000)

    # 🔥 mix behaviors
    if r < user["activity_level"]:
        return int(base * random.uniform(0.2, 1.5))
    else:
        return int(base * random.uniform(1.0, 3.0))

# COMMENT GENERATION
def generate_comment(user, post_time):
    text = sample_comment_text(user)

    delay = generate_delay(user)
    timestamp = post_time + timedelta(seconds=delay)

    return {
        "comment_id": str(uuid.uuid4()),
        "user_id": user["user_id"],
        "text": text,
        "timestamp": timestamp,
        "is_bot": user["is_bot"]
    }

# MAIN GENERATOR
def generate_data():
    users = [generate_user() for _ in range(NUM_USERS)]
    users_df = pd.DataFrame(users)

    comments = []
    posts = []

    for i in range(NUM_POSTS):
        post_id = f"post_{i}"
        post_time = datetime.now() - timedelta(days=random.randint(1, 30))

        posts.append({
            "post_id": post_id,
            "timestamp": post_time
        })

        for user in users:
            if random.random() < 0.2:
                comment = generate_comment(user, post_time)
                comment["post_id"] = post_id
                comments.append(comment)

    df_users = users_df.copy()


    # 🔥 LABEL NOISE (CRITICAL)

    noise_idx = df_users.sample(frac=0.01).index
    df_users.loc[noise_idx, "is_bot"] = 1 - df_users.loc[noise_idx, "is_bot"]

    return df_users, pd.DataFrame(posts), pd.DataFrame(comments)

# RUN
if __name__ == "__main__":
    users, posts, comments = generate_data()

    users.to_csv(GENERATED_DIR / "users.csv", index=False)
    posts.to_csv(GENERATED_DIR / "posts.csv", index=False)
    comments.to_csv(GENERATED_DIR / "comments.csv", index=False)

    print(f"✅ Data saved to: {GENERATED_DIR}")
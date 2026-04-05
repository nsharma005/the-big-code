import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# -------------------------
# LAZY LOAD SBERT
# -------------------------

_sbert_model = None

def get_sbert_model():
    global _sbert_model
    if _sbert_model is not None:
        return _sbert_model
    try:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        return _sbert_model
    except Exception as exc:
        print("Warning: sentence_transformers not available:", exc)
    _sbert_model = None
    return None

# -------------------------
# SPAM KEYWORDS
# -------------------------

SPAM_KEYWORDS = [
    "check out", "subscribe", "watch", "channel",
    "link", "visit", "free", "click", "youtube", "http", "www"
]

URL_REGEX = re.compile(r"http|www|youtube|\.com", re.IGNORECASE)

# -------------------------
# KEYWORD SCORE
# -------------------------

def spam_keyword_score(text):
    text = str(text).lower()
    return sum(1 for word in SPAM_KEYWORDS if word in text)

# -------------------------
# SAFE TEXT CLEANING
# -------------------------

def _clean_text(t):
    t = str(t).strip()
    if not t:
        return "empty"
    return t[:500]

# -------------------------
# MAIN FEATURE FUNCTION
# -------------------------

def comment_similarity_feature(df):
    df = df.copy()

    # -------------------------
    # BASIC FEATURES
    # -------------------------
    df["text"] = df["text"].astype(str).apply(_clean_text)

    df["text_length"] = df["text"].apply(len)

    df["has_url"] = df["text"].apply(lambda x: int(bool(URL_REGEX.search(x))))
    df["spam_score"] = df["text"].apply(spam_keyword_score)

    # extra strong features (kept same naming policy)
    df["num_exclamations"] = df["text"].str.count(r"!")
    df["num_caps"] = df["text"].str.count(r"[A-Z]")
    df["caps_ratio"] = df["num_caps"] / (df["text_length"] + 1e-5)

    # -------------------------
    # DUPLICATES
    # -------------------------
    df["duplicate_flag"] = df.duplicated(
        subset=["post_id", "text"]
    ).astype(int)

    # -------------------------
    # TF-IDF SIMILARITY (FULLY SAFE)
    # -------------------------
    df["avg_comment_similarity"] = 0.0

    for post_id, group in df.groupby("post_id"):

        if len(group) < 2:
            continue

        texts = group["text"].tolist()

        # ---- HARD GUARDS ----
        if len(texts) == 0:
            continue

        if all(len(t.strip()) == 0 for t in texts):
            df.loc[group.index, "avg_comment_similarity"] = 0.0
            continue

        if len(set(texts)) == 1:
            df.loc[group.index, "avg_comment_similarity"] = 1.0
            continue

        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=1500,
                ngram_range=(1, 2),
                min_df=2
            )

            X = vectorizer.fit_transform(texts)

            # 🚨 CRITICAL: empty feature space
            if X.shape[1] == 0:
                df.loc[group.index, "avg_comment_similarity"] = 0.0
                continue

            X = X.astype(np.float64)

            if hasattr(X, "data"):
                X.data = np.nan_to_num(
                    X.data, nan=0.0, posinf=0.0, neginf=0.0
                )

            X = normalize(X)

            # 🚨 check zero rows
            row_norms = np.sqrt((X.multiply(X)).sum(axis=1)).A1
            if np.all(row_norms == 0):
                df.loc[group.index, "avg_comment_similarity"] = 0.0
                continue

            # 🚨 safe cosine
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                sim_matrix = cosine_similarity(X)

            sim_matrix = np.nan_to_num(sim_matrix)

            avg_sim = sim_matrix.mean(axis=1)

            df.loc[group.index, "avg_comment_similarity"] = avg_sim

        except Exception as e:
            print(f"TF-IDF error in post {post_id}: {e}")
            df.loc[group.index, "avg_comment_similarity"] = 0.0

    # -------------------------
    # SBERT SIMILARITY (SAFE)
    # -------------------------
    df["semantic_similarity"] = 0.0

    model = get_sbert_model()

    if model is not None:
        for post_id, group in df.groupby("post_id"):

            if len(group) < 2:
                continue

            try:
                group_subset = group.head(100)

                texts = group_subset["text"].tolist()

                if len(texts) == 0:
                    continue

                embeddings = model.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )

                embeddings = np.nan_to_num(
                    embeddings, nan=0.0, posinf=0.0, neginf=0.0
                )

                norms = np.linalg.norm(embeddings, axis=1)

                if np.all(norms == 0) or embeddings.size == 0:
                    avg_sim = np.zeros(len(texts))
                else:
                    embeddings = normalize(embeddings)

                    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                        sim_matrix = cosine_similarity(embeddings)

                    sim_matrix = np.nan_to_num(sim_matrix)

                    avg_sim = sim_matrix.mean(axis=1)

                df.loc[group_subset.index, "semantic_similarity"] = avg_sim

            except Exception as e:
                print(f"SBERT error in post {post_id}: {e}")
                df.loc[group.index, "semantic_similarity"] = 0.0

    # -------------------------
    # NORMALIZATION
    # -------------------------
    df["spam_score"] = df["spam_score"] / (df["spam_score"].max() + 1e-5)
    df["text_length"] = np.log1p(df["text_length"])

    return df
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import community as community_louvain

def detect_bot_communities(G):
    """
    Detect communities using Louvain algorithm
    """

    if len(G.nodes) == 0:
        return []

    partition = community_louvain.best_partition(G)

    communities = {}

    for user, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(user)

    results = []

    for comm_id, users in communities.items():
        if len(users) > 3:   # filter small groups
            results.append({
                "community_id": comm_id,
                "users": users,
                "size": len(users)
            })

    return results

def build_user_graph(df):
    """
    Build graph where:
    - nodes = users
    - edges = similar commenting behavior
    """

    G = nx.Graph()

    users = df["user_id"].unique()
    G.add_nodes_from(users)

    # --- TEXT SIMILARITY ---
    user_texts = df.groupby("user_id")["text"].apply(lambda x: " ".join(x)).to_dict()

    vectorizer = TfidfVectorizer(max_features=1000)
    user_ids = list(user_texts.keys())
    text_corpus = list(user_texts.values())

    X = vectorizer.fit_transform(text_corpus)
    sim_matrix = cosine_similarity(X)

    # --- BUILD EDGES ---
    for i in range(len(user_ids)):
        for j in range(i + 1, len(user_ids)):

            if sim_matrix[i][j] > 0.7:  # threshold
                G.add_edge(user_ids[i], user_ids[j], weight=sim_matrix[i][j])

    # users commenting on same post → connect them
    for post_id, group in df.groupby("post_id"):
        users = group["user_id"].unique()
        
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                G.add_edge(users[i], users[j], weight=0.5)
    
    # users commenting within short time window
    for post_id, group in df.groupby("post_id"):
        group = group.sort_values("timestamp")

        for i in range(len(group) - 1):
            u1 = group.iloc[i]["user_id"]
            u2 = group.iloc[i+1]["user_id"]

            if abs(group.iloc[i]["time_since_post"] - group.iloc[i+1]["time_since_post"]) < 30:
                G.add_edge(u1, u2, weight=0.7)

    return G

def detect_bot_clusters(G):
    """
    Find connected components = potential bot groups
    """

    clusters = list(nx.connected_components(G))

    results = []

    for cluster in clusters:
        size = len(cluster)

        if size > 5:  # ignore tiny clusters
            results.append({
                "cluster_size": size,
                "users": list(cluster)
            })

    return results

def score_clusters(clusters, user_df):
    """
    Assign risk score to clusters
    """

    cluster_results = []

    for c in clusters:
        users = c["users"]

        sub = user_df[user_df["user_id"].isin(users)]

        avg_bot_prob = sub["bot_probability"].mean()

        cluster_results.append({
            "cluster_size": c["cluster_size"],
            "avg_bot_probability": round(avg_bot_prob, 2),
            "risk": "High" if avg_bot_prob > 0.6 else "Medium"
        })

    return pd.DataFrame(cluster_results)

def score_communities(communities, user_df):

    results = []

    for comm in communities:
        users = comm["users"]

        sub = user_df[user_df["user_id"].isin(users)]

        avg_bot_prob = sub["bot_probability"].mean()

        results.append({
            "community_id": comm["community_id"],
            "size": comm["size"],
            "avg_bot_probability": round(avg_bot_prob, 2),
            "risk": "High" if avg_bot_prob > 0.6 else "Medium"
        })

    return pd.DataFrame(results)
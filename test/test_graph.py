from evaluation_metrics.graph_detection import build_user_graph, detect_bot_communities


def test_graph_creation(sample_df):
    G = build_user_graph(sample_df)

    assert len(G.nodes) > 0


def test_community_detection(sample_df):
    G = build_user_graph(sample_df)
    communities = detect_bot_communities(G)

    assert isinstance(communities, list)
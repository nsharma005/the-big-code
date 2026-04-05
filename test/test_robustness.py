import numpy as np
from data_generator.generator_synthetic import generate_data


def test_data_generation():
    users, posts, comments = generate_data()

    assert len(users) > 0
    assert len(posts) > 0
    assert len(comments) > 0


def test_noise_stability():
    scores = []

    for _ in range(3):
        users, posts, comments = generate_data()

        # simple proxy: bot ratio
        scores.append(users["is_bot"].mean())

    assert np.std(scores) < 0.3  # stability check
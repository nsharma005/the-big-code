import time
from data_generator.generator_synthetic import generate_data


def test_large_scale_runtime():
    start = time.time()

    users, posts, comments = generate_data()

    end = time.time()

    assert (end - start) < 30  # generation must be fast
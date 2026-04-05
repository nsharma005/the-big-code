import sys
from pathlib import Path
import pytest
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))



# COMMON FIXTURES


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "user_id": ["u1", "u2"],
        "post_id": ["p1", "p1"],
        "text": ["nice video", "nice video"],
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "time_since_post": [10, 20]
    })


@pytest.fixture
def small_api_input():
    return {
        "users": [
            {
                "user_id": "u1",
                "followers": 100,
                "following": 50,
                "account_age": 200
            }
        ],
        "posts": [
            {
                "post_id": "p1",
                "timestamp": "2024-01-01T00:00:00"
            }
        ],
        "comments": [
            {
                "comment_id": "c1",
                "user_id": "u1",
                "post_id": "p1",
                "text": "Nice video!",
                "timestamp": "2024-01-01T00:01:00"
            }
        ]
    }
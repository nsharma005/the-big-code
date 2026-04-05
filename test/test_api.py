from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_api_response_structure(small_api_input):
    res = client.post("/predict", json=small_api_input)

    assert res.status_code == 200

    data = res.json()

    assert "users" in data
    assert "influencers" in data
    assert "clusters" in data


def test_api_empty_input():
    res = client.post("/predict", json={
        "users": [], "posts": [], "comments": []
    })

    assert res.status_code == 200
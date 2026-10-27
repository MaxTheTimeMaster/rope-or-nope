from fastapi.testclient import TestClient
from app_final import APP_VERSION, app
import pytest
from tasks import TASKS, TOKEN_TO_ID
import logging

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_version():
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json()["version"] == APP_VERSION

def test_root_points_to_docs():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"

def test_list_architectures():
    response = client.get("/architectures")
    assert response.status_code == 200
    names = {arch["name"] for arch in response.json()}
    assert {"learned", "sinusoidal", "rope", "alibi", "nope"} <= names


def test_get_architecture_by_name():
    response = client.get("/architectures/rope")
    assert response.status_code == 200
    assert response.json()["positions"] == "rope"


def test_unknown_architecture_returns_404():
    response = client.get("/architectures/gpt-67")
    assert response.status_code == 404


def test_reverse_sample_is_reversed():
    response = client.get("/tasks/reverse/sample", params={"length": 5, "seed": 0})
    assert response.status_code == 200
    data = response.json()
    assert len(data["input"]) == 5
    assert data["target"] == data["input"][::-1]


def test_sample_rejects_too_long_input():
    response = client.get("/tasks/copy/sample", params={"length": 1000})
    assert response.status_code == 422


def test_unknown_task_returns_404():
    response = client.get("/tasks/fizzbuzz/sample")
    assert response.status_code == 404

@pytest.mark.parametrize("name", list(TASKS))
def test_every_task_produces_known_tokens(name):
    response = client.get(f"/tasks/{name}/sample", params={"length": 6, "seed": 0})
    assert response.status_code == 200
    assert all(token in TOKEN_TO_ID for token in response.json()["target"])

def test_sample_uses_default_length_from_settings():
    from app_final import DEFAULT_SAMPLE_LENGTH

    response = client.get("/tasks/copy/sample")
    assert len(response.json()["input"]) == DEFAULT_SAMPLE_LENGTH


def test_requests_are_logged(caplog):
    with caplog.at_level(logging.INFO, logger="rope_or_nope"):
        client.get("/health")
    assert "GET /health -> 200" in caplog.text

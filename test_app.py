from fastapi.testclient import TestClient
from app_final import APP_VERSION, app

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

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_summarize_text():
    payload = {"text": "This is a short test document about Python programming. Python is widely used for data science and machine learning. It has a rich ecosystem of libraries."}
    response = client.post("/summarize/text", json=payload)
    assert response.status_code in [200, 503]
    # 503 is acceptable if AWS credentials not configured in CI
    if response.status_code == 200:
        data = response.json()
        assert "summary" in data
        assert "key_points" in data
        assert "prompt_version" in data


def test_summarize_invalid_file():
    # Send a non-PDF file
    files = {"file": ("test.txt", b"not a pdf", "text/plain")}
    response = client.post("/summarize", files=files)
    assert response.status_code == 400


def test_prompt_versions():
    response = client.get("/api/prompt-versions")
    assert response.status_code == 200
    versions = response.json()
    assert len(versions) >= 3
    version_ids = [v['version'] for v in versions]
    assert 'v1' in version_ids
    assert 'v2' in version_ids
    assert 'v3' in version_ids


def test_active_prompt():
    response = client.get("/api/active-prompt")
    assert response.status_code == 200
    assert "version" in response.json()
    assert "template" in response.json()


def test_eval_results_endpoint():
    response = client.get("/api/eval-results")
    assert response.status_code == 200

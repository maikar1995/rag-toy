import pytest
from fastapi.testclient import TestClient
from src.rag_toy.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_rag_ask(client):
    payload = {"query": "¿Qué es el principio de la pirámide de Minto?"}
    response = client.post("/api/v1/rag/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Chequeo de formato mínimo
    assert "answer" in data
    assert "citations" in data
    assert "confidence" in data
    assert "notes" in data
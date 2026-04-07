from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_predict_setosa():
    response = client.post("/predict", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    })
    assert response.status_code == 200
    assert "species" in response.json()


def test_predict_virginica():
    response = client.post("/predict", json={
        "sepal_length": 6.7,
        "sepal_width": 3.0,
        "petal_length": 5.2,
        "petal_width": 2.3,
    })
    assert response.status_code == 200
    assert response.json()["species"] == "Iris-virginica"


def test_predict_invalid():
    response = client.post("/predict", json={"sepal_length": "abc"})
    assert response.status_code == 422

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_request():
    payload = {
        "name": "Chanel Classic Flag Bag",
        "item_condition_id": 3,
        "category_name": "Women/Bags/Shoulder Bag",
        "brand_name": "Chanel",
        "shipping": 0,
        "item_description": "Barely used, comes with dust bag"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_price" in body
    assert "price_range_low" in body
    assert "price_range_high" in body
    assert body["price_range_low"] < body["predicted_price"] < body["price_range_high"]


def test_predict_invalid_condition_id():
    payload = {
        "name": "Test Item", "item_condition_id": 47,
        "category_name": "Women/Bags/Shoulder Bag", "shipping": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_required_field():
    payload = {
        "item_condition_id": 3,
        "category_name": "Women/Bags/Shoulder Bag", "shipping": 0
    }  # "name" missing
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_without_optional_fields():
    payload = {
        "name": "Generic Item", "item_condition_id": 3,
        "category_name": "Electronics/Phones", "shipping": 1
    }  # brand_name aur item_description dono missing
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_price" in response.json()
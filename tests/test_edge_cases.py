from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_unseen_brand_does_not_crash():
    """Day 1/10: brands never seen in training should fall back to the global mean,
    not crash -- this is the exact leak-safe encoding behavior we built."""
    payload = {
        "name": "Mystery Item", "item_condition_id": 3,
        "category_name": "Women/Bags/Shoulder Bag",
        "brand_name": "TotallyMadeUpBrandXYZ123",
        "shipping": 0, "item_description": "test"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["predicted_price"] > 0


def test_unseen_category_does_not_crash():
    """Day 3/6: OneHotEncoder(handle_unknown='ignore') should silently zero-encode
    a category never seen in training, not crash."""
    payload = {
        "name": "Mystery Item", "item_condition_id": 3,
        "category_name": "TotallyNewCategory/Nonexistent/Subcategory",
        "shipping": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_condition_boundary_values():
    """1 (New) and 5 (Poor) are the extreme valid values -- both must be
    accepted, not accidentally excluded by an off-by-one error."""
    base = {"name": "Test Item", "category_name": "Electronics/Phones", "shipping": 0}
    for condition in [1, 5]:
        response = client.post("/predict", json={**base, "item_condition_id": condition})
        assert response.status_code == 200, f"condition_id={condition} should be valid"


def test_condition_just_outside_boundary_rejected():
    """0 and 6 are one step outside 1-5 -- confirms the boundary sits
    exactly where expected, not off by one in either direction."""
    base = {"name": "Test Item", "category_name": "Electronics/Phones", "shipping": 0}
    for condition in [0, 6]:
        response = client.post("/predict", json={**base, "item_condition_id": condition})
        assert response.status_code == 422, f"condition_id={condition} should be rejected"


def test_empty_string_description():
    """An empty string differs from a missing field -- confirm it's handled
    like 'no description', not a crash on empty text."""
    payload = {
        "name": "Test Item", "item_condition_id": 3,
        "category_name": "Electronics/Phones", "shipping": 0,
        "item_description": ""
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
import pandas as pd

from src.feature_engineering import engineer_raw_features


def test_category_split_three_levels():
    df = pd.DataFrame(
        [
            {
                "name": "Test Item",
                "item_condition_id": 1,
                "category_name": "Women/Bags/Shoulder Bag",
                "brand_name": "Chanel",
                "shipping": 0,
                "item_description": "test desc",
            }
        ]
    )
    result = engineer_raw_features(df)
    assert result.loc[0, "main_category"] == "Women"
    assert result.loc[0, "sub_category"] == "Bags"
    assert result.loc[0, "sub_sub_category"] == "Shoulder Bag"


def test_missing_brand_defaults_to_no_brand():
    df = pd.DataFrame(
        [
            {
                "name": "Test Item",
                "item_condition_id": 1,
                "category_name": "Women/Bags/Shoulder Bag",
                "brand_name": None,
                "shipping": 0,
                "item_description": "test desc",
            }
        ]
    )
    result = engineer_raw_features(df)
    assert result.loc[0, "brand_name"] == "no brand"
    assert result.loc[0, "is_branded"] == 0


def test_missing_description_flagged_correctly():
    df = pd.DataFrame(
        [
            {
                "name": "Test Item",
                "item_condition_id": 1,
                "category_name": "Women/Bags/Shoulder Bag",
                "brand_name": "Chanel",
                "shipping": 0,
                "item_description": None,
            }
        ]
    )
    result = engineer_raw_features(df)
    assert result.loc[0, "has_description"] == 0
    assert result.loc[0, "item_description"] == "No description yet"


def test_condition_label_mapping():
    df = pd.DataFrame(
        [
            {
                "name": "Test Item",
                "item_condition_id": 1,
                "category_name": "Electronics/Phones",
                "brand_name": "Apple",
                "shipping": 1,
                "item_description": "test",
            }
        ]
    )
    result = engineer_raw_features(df)
    assert result.loc[0, "condition_label"] == "New"

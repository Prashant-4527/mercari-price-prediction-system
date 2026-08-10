import numpy as np
import pandas as pd


def engineer_raw_features(df):
    df = df.copy()

    cat_split = df["category_name"].str.split("/")
    df["main_category"] = cat_split.str[0]
    df["sub_category"] = cat_split.str[1]
    df["sub_sub_category"] = cat_split.str[2]
    df["category_depth"] = df["category_name"].str.split("/").str.len()

    df["name"] = df["name"].fillna("")
    df["item_description"] = df["item_description"].fillna("No description yet")
    df["brand_name"] = df["brand_name"].fillna("No Brand").str.lower().str.strip()

    df["name_length"] = df["name"].str.len()
    df["desc_length"] = df["item_description"].str.len()
    df["name_word_count"] = df["name"].str.split().str.len()
    df["has_description"] = df["item_description"].ne("No description yet").astype(int)
    df["is_branded"] = np.where(df["brand_name"] == "no brand", 0, 1)

    condition_map = {1: "New", 2: "Like New", 3: "Good", 4: "Fair", 5: "Poor"}
    df["condition_label"] = df["item_condition_id"].map(condition_map)

    return df


if __name__ == "__main__":
    sample_raw = pd.DataFrame([{
        "name": "Chanel Classic Flag Bag medium Caviar L",
        "item_condition_id": 3,
        "category_name": "Women/Bags/Shoulder Bag",
        "brand_name": "Chanel",
        "shipping": 0,
        "item_description": "Barely used, comes with dust bag"
    }])

    engineered = engineer_raw_features(sample_raw)
    print(engineered[["main_category", "sub_category", "sub_sub_category",
                       "condition_label", "name_length", "desc_length"]])
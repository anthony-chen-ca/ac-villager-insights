"""
featurize.py

Feature engineering module for Animal Crossing villager data.

This script constructs numerical feature vectors from structured villager data, including visual
CLIP embeddings (from villager icons and photos), categorical metadata (species, personality,
colors, etc.), and optionally applies PCA to reduce dimensionality for visual features.

Used by modeling scripts for prediction and recommendation tasks.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
from scripts.config import ModelConfig
from scripts.file_locations import MERGED_CSV


def parse_tags(s):
    return [tag.strip() for tag in s.split(";") if tag.strip()] if pd.notna(s) else []


def build_features(config: ModelConfig):
    """
    Builds a numerical feature matrix.

    This function extracts visual and categorical features. The resulting features are suitable
    for use in ML models.

    Returns:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Target vote count.
        feature_names (List[str]): Names of all features in the same order as X.
    """
    feature_names = []
    feature_parts = []
    df = pd.read_csv(MERGED_CSV)

    # Visual features
    for visual_feature in config.visual_settings:
        prefix = visual_feature.name.value + " CLIP"
        column_names = [col for col in df.columns if col.startswith(prefix)]
        column_data = df[column_names].values
        column_data = StandardScaler().fit_transform(column_data)

        # Apply PCA if requested
        if visual_feature.pca is not None:
            pca = PCA(
                n_components=visual_feature.pca,
                random_state=config.model_settings.random_seed,
            )
            column_data = pca.fit_transform(column_data)
            column_names = [f"{prefix} PCA {i}" for i in range(column_data.shape[1])]

        feature_names += column_names
        feature_parts.append(column_data)

    onehot_fields = []
    mlb_registry = {}

    # Categorical features
    for categorical_feature in config.categorical_settings:
        if categorical_feature == "Personality":
            # Combine Personality and Subtype into one field
            df["Personality Subtype"] = (
                df["Personality"].fillna("") + " " + df["Subtype"].fillna("")
            )
            onehot_fields.append("Personality Subtype")

        elif categorical_feature == "Style List":
            df["Style List"] = df[["Style 1", "Style 2"]].values.tolist()
            df["Style List"] = df["Style List"].apply(lambda lst: list(set(lst)))
            mlb = MultiLabelBinarizer()
            style_encoded = mlb.fit_transform(df["Style List"])
            mlb_registry["Style"] = mlb
            feature_parts.append(style_encoded)
            feature_names += [f"Style Tag {tag}" for tag in mlb.classes_]

        elif categorical_feature == "Color List":
            df["Color List"] = df[["Color 1", "Color 2"]].values.tolist()
            df["Color List"] = df["Color List"].apply(lambda lst: list(set(lst)))
            mlb = MultiLabelBinarizer()
            color_encoded = mlb.fit_transform(df["Color List"])
            mlb_registry["Color"] = mlb
            feature_parts.append(color_encoded)
            feature_names += [f"Color Tag {tag}" for tag in mlb.classes_]

        elif categorical_feature == "Meta Tags":
            df["Meta Tag List"] = df["Meta Tags"].apply(parse_tags)
            mlb = MultiLabelBinarizer()
            meta_encoded = mlb.fit_transform(df["Meta Tag List"])
            mlb_registry["Meta"] = mlb
            feature_parts.append(meta_encoded)
            feature_names += [f"Meta Tag {tag}" for tag in mlb.classes_]

        elif categorical_feature in [
            "Species",
            "Gender",
            "Hobby",
            "Favorite Song",
            "Default Umbrella",
            "Wallpaper",
            "Flooring",
            "Version Added",
            "Pocket Camp Theme",
        ]:
            onehot_fields.append(categorical_feature)

        else:
            raise ValueError(f"Unsupported categorical feature: {categorical_feature}")

    # One-hot encode all collected onehot_fields together
    if onehot_fields:
        cat_data = df[onehot_fields].fillna("Unknown")
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(cat_data)
        raw_feature_names = encoder.get_feature_names_out(onehot_fields)
        cleaned_feature_names = [name.replace("_", " ") for name in raw_feature_names]
        feature_parts.append(encoded)
        feature_names += cleaned_feature_names

    if not feature_parts:
        raise ValueError("No features selected.")

    # Combine all features
    X = np.concatenate(feature_parts, axis=1)

    assert X.shape[0] == len(df), "Mismatch between data rows and feature rows"

    y = df["Votes"].values

    return X, y, feature_names

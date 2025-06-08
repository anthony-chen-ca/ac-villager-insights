"""
featurize.py

Feature engineering module for Animal Crossing villager data.

This script constructs numerical feature vectors from structured villager data, including visual
CLIP embeddings (from villager icons and photos), categorical metadata (species, personality,
colors, etc.), and optionally applies PCA to reduce dimensionality for visual features.

Used by modeling scripts for prediction and recommendation tasks.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer


def build_features(
    df,
    icon_weight,
    photo_weight,
    use_categorical=True,
    n_components_icon=None,
    n_components_photo=None,
    random_state=None,
):
    """
    Transforms a villager DataFrame into a numerical feature matrix.

    This function extracts visual and categorical features, applies scaling, optional
    dimensionality reduction via PCA, and applies custom weighting to each feature group. The
    resulting features are suitable for use in ML models.

    Args:
        df (pd.DataFrame): Input DataFrame containing villager data.
        icon_weight (float): Weight multiplier for icon image CLIP features.
        photo_weight (float): Weight multiplier for photo image CLIP features.
        use_categorical (bool): Toggle all categorical features on and off.
        n_components_icon (int, optional): Number of PCA components for icon CLIP vectors. If
        None, PCA is skipped.
        n_components_photo (int, optional): Number of PCA components for photo CLIP vectors. If
        None, PCA is skipped.
        random_state (int, optional): Random seed for reproducibility of PCA.

    Returns:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Target vote count.
        feature_names (List[str]): Names of all features in the same order as X.
    """
    feature_names = []
    feature_parts = []
    df_local = df.copy()

    # Icon features
    if icon_weight > 0:
        icon_cols = [col for col in df_local.columns if col.startswith("Icon_CLIP_")]
        icon_data = df_local[icon_cols].values
        icon_scaled = StandardScaler().fit_transform(icon_data)

        if n_components_icon is not None:
            icon_pca = PCA(n_components=n_components_icon, random_state=random_state)
            icon_scaled = icon_pca.fit_transform(icon_scaled)
            icon_column_names = [f"Icon_PCA_{i}" for i in range(icon_scaled.shape[1])]
        else:
            icon_column_names = icon_cols

        icon_scaled *= icon_weight
        feature_names += icon_column_names
        feature_parts.append(icon_scaled)

    # Photo features
    if photo_weight > 0:
        photo_cols = [col for col in df_local.columns if col.startswith("Photo_CLIP_")]
        photo_data = df_local[photo_cols].values
        photo_scaled = StandardScaler().fit_transform(photo_data)

        if n_components_photo is not None:
            photo_pca = PCA(n_components=n_components_photo, random_state=random_state)
            photo_scaled = photo_pca.fit_transform(photo_scaled)
            photo_column_names = [f"Photo_PCA_{i}" for i in range(photo_scaled.shape[1])]
        else:
            photo_column_names = photo_cols

        photo_scaled *= photo_weight
        feature_names += photo_column_names
        feature_parts.append(photo_scaled)

    # Categorical features
    if use_categorical is True:
        # Combine Personality and Subtype
        df_local["PersonalitySubtype"] = (
            df_local["Personality"].fillna("") + "_" + df_local["Subtype"].fillna("")
        )

        # Combine Style List
        df_local["Style_List"] = df_local[["Style 1", "Style 2"]].values.tolist()
        df_local["Style_List"] = df_local["Style_List"].apply(lambda lst: list(set(lst)))

        # Combine Color List
        df_local["Color_List"] = df_local[["Color 1", "Color 2"]].values.tolist()
        df_local["Color_List"] = df_local["Color_List"].apply(lambda lst: list(set(lst)))

        # One-hot encoded feature list
        cat_data = df_local[
            [
                "Species",
                "Gender",
                "Personality",
                "PersonalitySubtype",
                "Hobby",
                "Style 1",
                "Style 2",
                "Color 1",
                "Color 2",
            ]
        ].fillna("Unknown")

        # Encode ordered features
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        cat_encoded = encoder.fit_transform(cat_data)
        cat_feature_names = encoder.get_feature_names_out(cat_data.columns)

        # Encode unordered set features
        mlb_style = MultiLabelBinarizer()
        mlb_color = MultiLabelBinarizer()
        style_list_encoded = mlb_style.fit_transform(df_local["Style_List"])
        color_list_encoded = mlb_color.fit_transform(df_local["Color_List"])

        # Concatenate all categorical parts and append everything
        cat_combined = np.concatenate([cat_encoded, style_list_encoded, color_list_encoded], axis=1)
        feature_parts.append(cat_combined)
        feature_names += list(cat_feature_names)
        feature_names += [f"Style_Tag_{s}" for s in mlb_style.classes_]
        feature_names += [f"Color_Tag_{c}" for c in mlb_color.classes_]

    if not feature_parts:
        raise ValueError("No features selected. Please readjust at least one weight from 0.")

    # Combine all features
    X = np.concatenate(feature_parts, axis=1)
    y = df_local["Votes"].values

    return X, y, feature_names

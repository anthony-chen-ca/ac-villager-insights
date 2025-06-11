"""
app.py

The front-end logic of the app.
"""

import streamlit as st
import math
from dataclasses import asdict
from scripts.modeling.ridge_model import run_ridge
from scripts.config import *


def categorical_feature_widget() -> List[str]:
    st.write("### Categorical Features")
    st.markdown(f"**Toggle All Below**")
    select_all = st.checkbox(f"Select All", key="categorical_select_all")
    cols = st.columns(2)
    num_rows = math.ceil(len(CATEGORICAL_FEATURE_LIST) / 2)
    selection = {}
    for col_idx in range(2):
        with cols[col_idx]:
            for row_idx in range(num_rows):
                idx = row_idx + col_idx * num_rows
                if idx < len(CATEGORICAL_FEATURE_LIST):
                    item = CATEGORICAL_FEATURE_LIST[idx]
                    selection[item] = st.checkbox(
                        item, value=select_all, key=f"categorical_{item}"
                    )
    categorical_settings = [
        item for item, is_selected in selection.items() if is_selected
    ]
    return categorical_settings


def handle_visual_feature(name: VisualType,
                          default_selected=False) -> Optional[VisualFeatureConfig]:
    use_feature = st.checkbox(
        name, value=default_selected, key=f"visual_{name.replace(' ', '_')}"
    )
    if use_feature:
        use_pca = st.checkbox(f"Apply PCA to {name.value}")
        n_components = None
        if use_pca:
            n_components = st.slider(
                f"PCA Components for {name.value}", min_value=1, max_value=256, value=8
            )
        return VisualFeatureConfig(name=name, pca=n_components)


def visual_feature_widget() -> List[VisualFeatureConfig]:
    st.write("### Visual Features")
    st.markdown(f"**Toggle All Below**")
    select_all = st.checkbox(f"Select All", key = "visual_select_all")
    left_column, right_column = st.columns(2)
    visual_settings = []

    with left_column:
        icon_feature_config = handle_visual_feature(VisualType.ICON, select_all)
        if icon_feature_config is not None:
            visual_settings.append(icon_feature_config)

    with right_column:
        photo_feature_config = handle_visual_feature(VisualType.PHOTO, select_all)
        if photo_feature_config is not None:
            visual_settings.append(photo_feature_config)

    return visual_settings


def model_widget():
    model = st.selectbox("Model", [ModelType.RIDGE.value, ModelType.RANDOM_FOREST.value])
    alpha, n_estimators, max_depth, random_seed = None, None, None, None

    if model == ModelType.RIDGE:
        alpha = st.slider("Ridge Alpha", value=1.0, step=0.1)
    elif model == ModelType.RANDOM_FOREST:
        n_estimators = st.slider("Number of Estimators", value=1, step=1)
        max_depth = st.slider("Max Depth")

    use_random = st.checkbox("Use Random Seed", key="random")
    if use_random:
        random_seed = st.slider("Random Seed", min_value=0, max_value=100, value=42, step=1)

    return ModelSettings(model=model,
                         alpha=alpha,
                         n_estimators=n_estimators,
                         max_depth=max_depth,
                         random_seed=random_seed)


def main():
    st.set_page_config(page_title="ACNH Popularity Predictor", page_icon="🍃")

    st.title("Animal Crossing Villager Popularity Prediction")
    st.header("Model Configuration")

    st.subheader("Step 1: Choose your features")
    categorical_settings = categorical_feature_widget()
    visual_settings = visual_feature_widget()

    st.subheader("Step 2: Choose your model and hyperparameters")
    model_settings = model_widget()

    config = ModelConfig(
        categorical_settings=categorical_settings,
        visual_settings=visual_settings,
        model_settings=model_settings
    )

    st.subheader("Step 3: Run the model!")
    if st.button("Run Model"):
        try:
            config.validate()
            st.json(asdict(config))
            if config.model_settings.model == ModelType.RIDGE:
                results = run_ridge(config)

                st.metric("MAE", f"{results['mae']:.2f}")
                st.metric("R²", f"{results['r2']:.3f}")

                st.subheader("Predictions on Test Set")
                st.dataframe(results["test_results"])

                st.subheader("Top 20 Most Influential Features")
                st.dataframe(results["top_coefficients"][["Feature", "Coefficient"]])

                st.image(results["coef_plot_path"],
                         caption="Top 20 Coefficients (Ridge)")
        except ValueError as e:
            st.error(str(e))


if __name__ == "__main__":
    main()

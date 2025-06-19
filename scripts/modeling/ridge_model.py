"""
ridge_model.py
"""

import pandas as pd
import numpy as np
import plotly.express as px
import os
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scripts.config import ModelConfig
from scripts.file_locations import MERGED_CSV, OUTPUTS_DIR
from scripts.featurize import build_features


def run_ridge(config: ModelConfig):
    """
    Trains a Ridge regression model on villager features and evaluates performance.

    Args:
        config (ModelConfig): Configuration containing model and feature parameters.

    Returns:
        dict: A dictionary containing:
            - 'mae': Mean Absolute Error on test set
            - 'r2': R^2 score on test set
            - 'test_results': pd.DataFrame with columns [Name, Predicted, Actual]
            - 'top_coefficients': pd.DataFrame with columns [Feature, Coefficient]

    """
    # Config
    alpha = config.model_settings.alpha
    seed = config.model_settings.random_seed
    top_k = 20

    # Load features
    X, y_raw, feature_names = build_features(config)

    # Target transformation
    y = np.log1p(y_raw)

    # Recover full name list for display
    df = pd.read_csv(MERGED_CSV)
    names = df["Name"].values

    # Track row indices
    df["row_idx"] = df.index

    # Train/test split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df["row_idx"].values, test_size=0.2, random_state=seed
    )

    # Model training
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    # Predictions
    y_pred_log = model.predict(X_test)
    y_pred_raw = np.expm1(y_pred_log)
    y_true_raw = y_raw[idx_test]

    # Evaluation metrics
    mae = mean_absolute_error(y_true_raw, y_pred_raw)
    r2 = r2_score(y_true_raw, y_pred_raw)

    # Test prediction output
    test_results = pd.DataFrame({
        "Image": df.loc[idx_test, "Icon Image URL"].values,
        "Name": names[idx_test],
        "Predicted": np.round(y_pred_raw).astype(int),
        "Actual": y_true_raw.astype(int)
    }).sort_values("Predicted", ascending=False)

    # Coefficient analysis
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_,
    })
    coef_df["AbsCoefficient"] = coef_df["Coefficient"].abs()
    top_coefficients = coef_df.sort_values("AbsCoefficient", ascending=False).head(top_k)

    # Save top-K coefficient plot
    coef_plot_path = os.path.join(OUTPUTS_DIR, "top_k_coefficients.png")

    # Create Plotly bar plot
    fig = px.bar(
        top_coefficients[::-1],  # reverse for descending vertical
        x="Coefficient",
        y="Feature",
        orientation="h",
        title=f"Top {top_k} Most Influential Features (Ridge)",
        text="Coefficient"
    )
    fig.update_layout(yaxis=dict(tickfont=dict(size=10)))
    fig.update_traces(marker_color='mediumseagreen')  # Optional styling

    # Save as image path only if you want to keep it (optional now)
    fig.write_image(coef_plot_path)

    return {
        "mae": mae,
        "r2": r2,
        "test_results": test_results,
        "top_coefficients": top_coefficients,
        "coef_plot": fig,
    }

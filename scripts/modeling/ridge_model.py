"""
ridge_model.py

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scripts.config import ModelConfig
from scripts.file_locations import MERGED_CSV, OUTPUTS_DIR
from scripts.featurize import build_features


def run_ridge(config: ModelConfig):
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

    plt.figure(figsize=(10, 6))
    plt.barh(top_coefficients["Feature"][::-1], top_coefficients["Coefficient"][::-1])
    plt.xlabel("Coefficient")
    plt.title(f"Top {top_k} Most Influential Features (Ridge)")
    plt.tight_layout()
    plt.savefig(coef_plot_path, bbox_inches="tight")
    plt.close()

    return {
        "mae": mae,
        "r2": r2,
        "test_results": test_results,
        "top_coefficients": top_coefficients,
        "coef_plot_path": coef_plot_path,
    }

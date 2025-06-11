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
from scripts.file_locations import MERGED_CSV, OUTPUTS_DIR
from scripts.featurize import build_features


def run_ridge(config: dict):
    # PARAMETERS
    icon_weight = 1.0
    photo_weight = 1.0
    use_categorical = True
    n_components_icon = 8
    n_components_photo = 8
    alpha = 1.0
    random_state = 41  # For reproducibility across runs

    # OUTPUT PARAMETERS
    save_test_results = True
    save_top_k_coefficients = True
    k = 25

    print("===== MODEL SETUP =====")
    print(f"icon_weight = {icon_weight}")
    print(f"photo_weight = {photo_weight}")
    print(f"use_categorical = {use_categorical}")
    print(f"n_components_icon = {n_components_icon}")
    print(f"n_components_photo = {n_components_photo}")
    print(f"alpha = {alpha}")

    # Load merged dataset
    df = pd.read_csv(MERGED_CSV)

    # Build X, y with CLIP + categorical features
    X, y_raw, feature_names = build_features(
        df,
        icon_weight,
        photo_weight,
        use_categorical,
        n_components_icon,
        n_components_photo,
        random_state,
    )

    # Log-transform target
    y = np.log1p(y_raw)

    # Track indices and names
    df["row_idx"] = df.index
    names = df["Name"].values

    # Split the data into training and test datasets
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df["row_idx"].values, test_size=0.2, random_state=random_state
    )

    # Train the model
    model = Ridge(alpha)
    model.fit(X_train, y_train)

    # Predict in log space
    y_pred_log = model.predict(X_test)
    y_pred_raw = np.expm1(y_pred_log)
    y_true_raw = y_raw[idx_test]

    # Evaluate and output results
    mae = mean_absolute_error(y_true_raw, y_pred_raw)
    r2 = r2_score(y_true_raw, y_pred_raw)

    print("===== MODEL RESULTS =====")
    print(f"Mean Absolute Error (Votes): {mae:.2f}")
    print(f"R² Score: {r2:.2f}")

    # ================================================================================
    # SAVE PREDICTED TEST RESULTS
    # ================================================================================
    if save_test_results:
        # Create test prediction DataFrame
        prediction_results = pd.DataFrame(
            {
                "Name": names[idx_test],
                "Predicted": np.round(y_pred_raw).astype(int),
                "Actual": y_true_raw.astype(int),
            }
        )

        # Sort by predicted votes descending
        prediction_results_sorted = prediction_results.sort_values("Predicted", ascending=False)

        # Output path
        output_path = os.path.join(OUTPUTS_DIR, "test_predictions.csv")
        prediction_results_sorted.to_csv(output_path, index=False)

        print(f"\nTest predictions saved to {output_path}")

    # ================================================================================
    # TOP K COEFFICIENTS
    # ================================================================================
    if save_top_k_coefficients:
        # Get coefficients and sort them
        coefficient_df = pd.DataFrame({"Feature": feature_names, "Coefficient": model.coef_})

        # Sort by absolute value to find the most influential
        coefficient_df["AbsCoefficient"] = np.abs(coefficient_df["Coefficient"])
        top_features = coefficient_df.sort_values("AbsCoefficient", ascending=False).head(k)

        plt.figure(figsize=(10, 6))
        plt.barh(top_features["Feature"][::-1], top_features["Coefficient"][::-1])
        plt.xlabel("Coefficient")
        plt.title(f"Top {k} Most Influential Features (Ridge)")
        plt.tight_layout()

        global_path = os.path.join(OUTPUTS_DIR, "top_k_coefficients.png")
        plt.savefig(global_path, bbox_inches="tight")
        plt.close()
        print(f"Top {k} coefficients saved to {global_path}")

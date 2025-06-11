"""
hyperparameter_tuning.py

Tunes hyperparameters for the prediction model.
"""

import pandas as pd
import numpy as np
from itertools import product
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scripts.file_locations import MERGED_CSV, HYPERPARAMETER_RESULTS_CSV
from scripts.featurize import build_features


def hyperparameter_sweep():
    # Load merged dataset
    df = pd.read_csv(MERGED_CSV)
    y_raw = df["Votes"].values
    df["row_idx"] = df.index
    names = df["Name"].values

    # Fixed parameters
    use_categorical = True
    random_state = 42
    test_size = 0.2

    # Parameter ranges to sweep
    icon_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    photo_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    icon_components = [None, 8, 16, 32]
    photo_components = [None, 8, 16, 32]
    alphas = [0.1, 0.5, 1.0, 2.0, 10.0]

    results = []

    # Loop over all combinations
    for iw, pw, ic, pc, alpha in product(
        icon_weights, photo_weights, icon_components, photo_components, alphas
    ):
        # Skip unnecessary icon component configurations
        if iw == 0.0 and ic is not None:
            continue

        # Skip unnecessary photo component configurations
        if pw == 0.0 and pc is not None:
            continue

        # Build features
        X, y, _ = build_features(df, iw, pw, use_categorical, ic, pc, random_state)
        y_log = np.log1p(y)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_log, test_size=test_size, random_state=random_state
        )

        # Train model
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        # Evaluate and save results
        y_pred = np.expm1(model.predict(X_test))
        y_true = np.expm1(y_test)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        results.append(
            {
                "icon_weight": iw,
                "photo_weight": pw,
                "n_components_icon": ic,
                "n_components_photo": pc,
                "alpha": alpha,
                "MAE": mae,
                "R2": r2,
            }
        )
        print(
            f"Done: iw={iw}, pw={pw}, ic={ic}, pc={pc}, alpha={alpha} -> MAE={mae:.2f}, R2={r2:.2f}"
        )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by MAE (ascending) and R² (descending)
    best_mae_row = results_df.sort_values("MAE", ascending=True).iloc[0]
    best_r2_row = results_df.sort_values("R2", ascending=False).iloc[0]

    print("\n=== Best by Lowest MAE ===")
    print(best_mae_row)
    print("\n=== Best by Highest R² ===")
    print(best_r2_row)

    # Save to CSV
    results_df.to_csv(HYPERPARAMETER_RESULTS_CSV, index=False)
    print(f"\nAll results saved to {HYPERPARAMETER_RESULTS_CSV}")


def retrieve_results():
    # Load results
    df = pd.read_csv(HYPERPARAMETER_RESULTS_CSV)

    # Sort by MAE (ascending) and R2 (descending)
    best_mae_row = df.sort_values("MAE", ascending=True).iloc[0]
    best_r2_row = df.sort_values("R2", ascending=False).iloc[0]

    print("=== Best by Lowest MAE ===")
    print(best_mae_row)
    print("\n=== Best by Highest R² ===")
    print(best_r2_row)


run_hyperparameter_testing = False


if __name__ == "__main__":
    if run_hyperparameter_testing is True:
        hyperparameter_sweep()
    else:
        retrieve_results()

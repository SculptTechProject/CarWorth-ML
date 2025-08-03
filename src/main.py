import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
import joblib

from .pipeline import build_pipeline
from ._helpers import (
    _ensure_dir,
    plot_parity,
    plot_residuals,
    plot_hist,
    plot_perm_importance,
    find_target_column,
)

# === Helper plotting functions ===


def main(args):
    """Main function to load data, train model and evaluate."""
    print("[1/5] Loads data from:", args.data)
    df = pd.read_csv(args.data)
    target_col = find_target_column(df)
    print(f"[2/5] Target column: {target_col}")

    y = df[target_col]
    X = df.drop(columns=[c for c in [target_col, "id"] if c in df.columns])

    # Auto-detect numeric/categorical columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    print(f"[3/5] Num: {len(num_cols)} | Cat: {len(cat_cols)}")

    model = build_pipeline(num_cols, cat_cols)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    print("[4/5] Training model…")
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    mae = mean_absolute_error(y_te, pred)
    rmse = mean_squared_error(y_te, pred) ** 0.5
    print(f"[5/5] Results → MAE: {mae:,.0f} | RMSE: {rmse:,.0f}")

    # Optional reports
    reports_dir = "reports"
    if getattr(args, "plots", False):
        _ensure_dir(reports_dir)
        plot_parity(
            y_te,
            pred,
            os.path.join(reports_dir, "parity_test.png"),
            title="Parity (test)",
        )
        plot_residuals(
            y_te,
            pred,
            os.path.join(reports_dir, "residuals_test.png"),
            title="Residuals vs Predicted (test)",
        )
        plot_hist(
            y_te - pred,
            os.path.join(reports_dir, "residuals_hist_test.png"),
            title="Residuals histogram (test)",
        )
        try:
            plot_perm_importance(
                model,
                X_tr,
                y_tr,
                os.path.join(reports_dir, "perm_importance_test.png"),
                top_n=20,
            )
        except Exception as e:
            print("[WARN] Permutation importance failed:", e)
        print("[PLOTS] Saved plots to:", reports_dir)

    if getattr(args, "cv", False):
        print("[CV] Computing 5-fold OOF predictions…")
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
        oof_mae = mean_absolute_error(y, oof_pred)
        oof_rmse = mean_squared_error(y, oof_pred) ** 0.5
        print(f"[CV] OOF → MAE: {oof_mae:,.0f} | RMSE: {oof_rmse:,.0f}")
        if getattr(args, "plots", False):
            plot_parity(
                y,
                oof_pred,
                os.path.join(reports_dir, "parity_oof.png"),
                title="Parity (5-fold OOF)",
            )
            plot_residuals(
                y,
                oof_pred,
                os.path.join(reports_dir, "residuals_oof.png"),
                title="Residuals vs Predicted (5-fold OOF)",
            )
            plot_hist(
                y - oof_pred,
                os.path.join(reports_dir, "residuals_hist_oof.png"),
                title="Residuals histogram (5-fold OOF)",
            )

    # Save the model
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        joblib.dump(model, args.out)
        print("[--SAVED MODEL--] Model saved in:", args.out)
        print("============= Success! =============")
    else:
        print("[--ERROR--] No output path provided. Model not saved.")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/car_dataset.csv",
        help="Path to CSV with target column e.g. 'price_pln' / 'price'.",
    )
    parser.add_argument(
        "--out",
        default="models/model.joblib",
        help="The save path of the trained model.",
    )
    parser.add_argument(
        "--plots", action="store_true", help="Save plots into reports/ directory."
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run 5-fold cross-validation and report OOF metrics.",
    )
    args = parser.parse_args()
    # Run the main func with parsed arguments
    main(args)

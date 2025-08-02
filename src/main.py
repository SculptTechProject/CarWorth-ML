import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.inspection import permutation_importance
import joblib



def build_pipeline(num_cols, cat_cols):
    """Preprocessing (impute + OHE) + HistGradientBoosting wrapped with log target."""
    preprocessor = ColumnTransformer(
        transformers=[
            # Numeric columns: impute missing values with median
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                # Categorical columns: impute missing values with most frequent and OHE
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "ohe",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=20,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    # HistGradientBoostingRegressor with log target transformation
    reg = HistGradientBoostingRegressor(random_state=42)
    return Pipeline(
        [
            ("prep", preprocessor),
            (
                "reg",
                TransformedTargetRegressor(
                    regressor=reg, func=np.log1p, inverse_func=np.expm1
                ),
            ),
        ]
    )


# === Helper plotting functions ===

def _ensure_dir(path: str):
    """Ensure that the directory exists."""
    os.makedirs(path, exist_ok=True)


def plot_parity(y_true, y_pred, out_path: str, title: str = "Parity plot"):
    """Plot true vs predicted values."""
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.4)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims)
    plt.xlabel("True price [PLN]")
    plt.ylabel("Predicted price [PLN]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_residuals(y_true, y_pred, out_path: str, title: str = "Residuals vs Predicted"):
    """Plot residuals (y - ŷ) vs predicted values."""
    resid = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, resid, alpha=0.4)
    plt.axhline(0)
    plt.xlabel("Predicted price [PLN]")
    plt.ylabel("Residual (y - ŷ) [PLN]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_hist(residuals, out_path: str, title: str = "Residuals histogram"):
    """Plot histogram of residuals (y - ŷ)."""
    plt.figure()
    plt.hist(residuals, bins=40)
    plt.xlabel("Residual (y - ŷ) [PLN]")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_perm_importance(model, X_te, y_te, out_path: str, top_n: int = 20):
    """Plot permutation feature importance."""
    r = permutation_importance(model, X_te, y_te, scoring="neg_mean_absolute_error", n_repeats=5, random_state=42)
    feat_names = model.named_steps["prep"].get_feature_names_out()
    order = r.importances_mean.argsort()[::-1][:top_n]
    plt.figure(figsize=(8, max(3, int(top_n * 0.3))))
    plt.barh(range(len(order)), r.importances_mean[order])
    plt.yticks(range(len(order)), [feat_names[i] for i in order])
    plt.gca().invert_yaxis()
    plt.xlabel("Permutation importance (ΔMAE)")
    plt.title("Top features")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


essential_targets = ["price_pln", "price", "msrp"]


def find_target_column(df):
    """Find the target column in the DataFrame."""
    for t in essential_targets:
        if t in df.columns:
            return t
    raise ValueError(
        f"Can't find proper columns. Searched: {essential_targets}. You have: {list(df.columns)}"
    )


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
        plot_parity(y_te, pred, os.path.join(reports_dir, "parity_test.png"), title="Parity (test)")
        plot_residuals(y_te, pred, os.path.join(reports_dir, "residuals_test.png"), title="Residuals vs Predicted (test)")
        plot_hist(y_te - pred, os.path.join(reports_dir, "residuals_hist_test.png"), title="Residuals histogram (test)")
        try:
            plot_perm_importance(model, X_tr, y_tr, os.path.join(reports_dir, "perm_importance_test.png"), top_n=20)
        except Exception as e:
            print("[WARN] Permutation importance failed:", e)
        print("[PLOtS] Saved plots to:", reports_dir)

    if getattr(args, "cv", False):
        print("[CV] Computing 5-fold OOF predictions…")
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
        oof_mae = mean_absolute_error(y, oof_pred)
        oof_rmse = mean_squared_error(y, oof_pred) ** 0.5
        print(f"[CV] OOF → MAE: {oof_mae:,.0f} | RMSE: {oof_rmse:,.0f}")
        if getattr(args, "plots", False):
            plot_parity(y, oof_pred, os.path.join(reports_dir, "parity_oof.png"), title="Parity (5-fold OOF)")
            plot_residuals(y, oof_pred, os.path.join(reports_dir, "residuals_oof.png"), title="Residuals vs Predicted (5-fold OOF)")
            plot_hist(y - oof_pred, os.path.join(reports_dir, "residuals_hist_oof.png"), title="Residuals histogram (5-fold OOF)")

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
    parser.add_argument("--plots", action="store_true", help="Save plots into reports/ directory.")
    parser.add_argument("--cv", action="store_true", help="Run 5-fold cross-validation and report OOF metrics.")
    args = parser.parse_args()
    # Run the main func with parsed arguments
    main(args)

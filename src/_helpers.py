import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import os


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


def plot_residuals(
    y_true, y_pred, out_path: str, title: str = "Residuals vs Predicted"
):
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
    r = permutation_importance(
        model,
        X_te,
        y_te,
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=42,
    )
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

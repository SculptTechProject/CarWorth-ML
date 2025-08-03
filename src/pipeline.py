from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np


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

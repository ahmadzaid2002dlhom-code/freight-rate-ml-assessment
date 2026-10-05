"""Fold-fitted sklearn preprocessing with safe unknown-category handling."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURE_COLUMNS, NUMERIC_FEATURE_COLUMNS


def build_sklearn_preprocessor() -> ColumnTransformer:
    """Return an unfitted processor for the shared engineered feature schema.

    Fit this inside the model pipeline on its training fold only. Unknown labels
    become all-zero blocks in their categorical feature, while numeric geography
    and other measurements remain available. Empty numeric columns are retained.
    """
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURE_COLUMNS),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True),
         CATEGORICAL_FEATURE_COLUMNS),
    ], remainder="drop", sparse_threshold=1.0)

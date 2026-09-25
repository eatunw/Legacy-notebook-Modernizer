"""
model.py — Training script for the Titanic survival pipeline.

Fixes addressed
---------------
ISSUE-02 : 80/20 train/validation split; validation accuracy is printed.
ISSUE-03 : Entire pipeline is serialised to app/artifacts/pipeline.joblib.
ISSUE-04 : OneHotEncoder fitted on training data only, embedded in a Pipeline,
           so encoding is always aligned at inference time.
ISSUE-05 : File I/O wrapped in try/except with informative error messages.
ISSUE-06 : Logic split into load_data(), build_pipeline(), train(), predict().
ISSUE-08 : random_state=1 used throughout (matches original notebook).
"""

import pathlib
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ── reproducibility (ISSUE-08) ───────────────────────────────────────────────
RANDOM_STATE = 1
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR = pathlib.Path("data")
ARTIFACTS_DIR = pathlib.Path("app/artifacts")
PIPELINE_PATH = ARTIFACTS_DIR / "pipeline.joblib"

# ── features (identical to original notebook) ────────────────────────────────
FEATURES = ["Pclass", "Sex", "SibSp", "Parch"]
TARGET = "Survived"
CATEGORICAL_FEATURES = ["Sex"]
NUMERIC_FEATURES = ["Pclass", "SibSp", "Parch"]


# ── ISSUE-06: modular functions ──────────────────────────────────────────────

def load_data(path: pathlib.Path) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame.  Raises on missing file or columns."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found: {path}")
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc
    return df


def build_pipeline() -> Pipeline:
    """
    Build the sklearn Pipeline that combines preprocessing and the classifier.

    Preprocessing (ISSUE-04):
        - OneHotEncoder on the 'Sex' column, fitted on training data only.
          handle_unknown='ignore' ensures unseen values at inference time
          produce an all-zeros row rather than an error.
        - Numeric features pass through unchanged (they are already integers).

    Model:
        - RandomForestClassifier with the exact hyper-parameters from the
          original notebook (n_estimators=100, max_depth=5, random_state=1).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="passthrough",  # numeric columns pass through unchanged
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return pipeline


def train(df: pd.DataFrame) -> tuple[Pipeline, float]:
    """
    Split into train/validation sets, fit the pipeline, and return it together
    with the validation accuracy (ISSUE-02).
    """
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    val_accuracy = pipeline.score(X_val, y_val)
    return pipeline, val_accuracy


def predict(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted_labels, probabilities_of_class_1)."""
    labels = pipeline.predict(X)
    proba = pipeline.predict_proba(X)[:, 1]
    return labels, proba


def save_pipeline(pipeline: Pipeline, path: pathlib.Path = PIPELINE_PATH) -> None:
    """Serialise the entire fitted pipeline to disk (ISSUE-03)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Pipeline saved: {path}")


def load_pipeline(path: pathlib.Path = PIPELINE_PATH) -> Pipeline:
    """Load a previously serialised pipeline from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"No serialised pipeline found at {path}. Run model.py first."
        )
    return joblib.load(path)


# ── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading training data …")
    train_df = load_data(DATA_DIR / "titanic_train.csv")

    print(f"Training rows: {len(train_df)}")
    pipeline, val_accuracy = train(train_df)

    print(f"\nValidation accuracy : {val_accuracy:.4f}  ({val_accuracy * 100:.2f} %)")

    save_pipeline(pipeline)
    print("Training complete.")

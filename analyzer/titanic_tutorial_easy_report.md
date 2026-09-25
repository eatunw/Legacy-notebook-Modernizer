# Titanic Tutorial Easy — Modernization Analysis Report

**Source notebook:** `sample_notebooks/titanic_tutorial_easy.ipynb`  
**Total cells:** 15 (8 markdown, 5 code, 2 markdown-only outro)  
**Analysis scope:** Production-readiness for a FastAPI service

---

## 1. Plain-English Explanation

### What the notebook does, end to end

This notebook is a beginner-oriented Kaggle walkthrough for the Titanic binary classification competition. It proceeds in four logical phases:

**Phase 1 — Environment probe (Cell 2)**  
Walks the Kaggle input directory tree using `os.walk` and prints every file path under `/kaggle/input`. This is pure Kaggle scaffolding; it produces no artefact used later.

**Phase 2 — Data loading (Cells 4, 6)**  
Reads two CSV files from hardcoded Kaggle-platform paths:
- `train.csv` → `train_data` (891 rows, includes the `Survived` label)
- `test.csv` → `test_data` (418 rows, no label — the competition's holdout set)

No validation, dtype coercion, or null-handling is performed at load time.

**Phase 3 — Exploratory survival-rate analysis (Cells 8, 10)**  
Computes two scalar statistics from the *training* data only:
- Percentage of female passengers who survived (`~74 %`)
- Percentage of male passengers who survived (`~19 %`)

These are printed to stdout and used to motivate the gender-based baseline. No intermediate output is persisted.

**Phase 4 — Model training and inference (Cell 12)**  
A `RandomForestClassifier` (100 trees, max depth 5, `random_state=1`) is trained on four features selected from `train_data`:
- `Pclass`, `Sex`, `SibSp`, `Parch`

Categorical encoding is handled by `pd.get_dummies` applied independently to `train_data[features]` and `test_data[features]`. The fitted model is then used to predict survival for every row in `test_data`, and results are written to `submission.csv` in the working directory.

### Core logic that must survive a refactor

The following elements carry the actual business logic and must be preserved faithfully:

| Step | Logic | Cell |
|---|---|---|
| Feature selection | `["Pclass", "Sex", "SibSp", "Parch"]` | 12 |
| Encoding | `pd.get_dummies` (one-hot, drop-first not applied) | 12 |
| Model | `RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)` | 12 |
| Fit / predict contract | fit on `(X, y)`, predict on `X_test` | 12 |
| Output schema | `PassengerId` + `Survived` columns in a CSV | 12 |
| Survival-rate statistics | gender breakdown computation | 8, 10 |

Everything else (file-walker, markdown prose, `.head()` display calls) is tutorial scaffolding and can be discarded.

---

## 2. Issues List

### ISSUE-01 · Hardcoded Kaggle-platform absolute paths

| Attribute | Detail |
|---|---|
| **Severity** | 🔴 CRITICAL |
| **Cells** | Cell 2 (`os.walk('/kaggle/input')`), Cell 4 (`"/kaggle/input/titanic/train.csv"`), Cell 6 (`"/kaggle/input/titanic/test.csv"`) |
| **Description** | All three data-access statements use absolute paths that are hard-wired to the Kaggle hosted runtime. Running this notebook on any other machine — including the target FastAPI server — raises `FileNotFoundError` immediately. |
| **Fix** | Replace with relative paths or environment-variable-driven config (e.g. `DATA_DIR = os.getenv("DATA_DIR", "data/")`); inject paths through a settings object in the FastAPI layer. |

---

### ISSUE-02 · No train/validation split — full training set used for both fitting and implicit evaluation

| Attribute | Detail |
|---|---|
| **Severity** | 🔴 CRITICAL |
| **Cell** | Cell 12 |
| **Description** | The model is trained on 100 % of `train_data` with no held-out validation set. There is no metric (accuracy, ROC-AUC, F1) computed against unseen data. The only "test" set is the unlabelled competition holdout, which cannot be used for quality control. In a production service this means there is no automated gate on model quality before deployment. |
| **Fix** | Add `sklearn.model_selection.train_test_split` to carve out at least 20 % as a local validation set and compute at minimum an accuracy score before serialising the model. |

---

### ISSUE-03 · No model serialisation / artefact persistence

| Attribute | Detail |
|---|---|
| **Severity** | 🔴 CRITICAL |
| **Cell** | Cell 12 |
| **Description** | The trained `RandomForestClassifier` object exists only in notebook memory. It is never saved to disk (e.g. via `joblib.dump` or `pickle`). A FastAPI endpoint cannot load or serve a model that was never persisted; the notebook would have to be re-executed on every server restart. |
| **Fix** | Add `joblib.dump(model, "models/titanic_rf.joblib")` immediately after `model.fit`, and load it in the FastAPI startup event with `joblib.load`. |

---

### ISSUE-04 · Data leakage risk from independent `get_dummies` calls

| Attribute | Detail |
|---|---|
| **Severity** | 🔴 CRITICAL |
| **Cell** | Cell 12, lines 5–6 |
| **Description** | `pd.get_dummies` is called separately on `train_data[features]` and `test_data[features]`. If any categorical level appears in one split but not the other, the resulting column sets differ silently. In practice, if `test_data` is missing a category (e.g. a rare `Pclass` value), `model.predict(X_test)` will raise a `ValueError` or, worse, silently produce incorrect predictions if column counts happen to match by coincidence. In a production API, the test data is a single row per request — the divergence is near-certain. |
| **Fix** | Fit a `sklearn.preprocessing.OneHotEncoder` (or a `ColumnTransformer` + `Pipeline`) on training data only, then `transform` at inference time to guarantee identical column layout. |

---

### ISSUE-05 · No error handling anywhere

| Attribute | Detail |
|---|---|
| **Severity** | 🔴 CRITICAL |
| **Cells** | Cells 4, 6, 8, 10, 12 |
| **Description** | There are no `try/except` blocks, no null-value guards, no type-validation checks, and no assertions. Silent failures that would crash a production service include: CSV file not found, required columns absent from input, unexpected `NaN` values in `Pclass`/`SibSp`/`Parch`, non-string values in the `Sex` column, and empty DataFrames. |
| **Fix** | Wrap file I/O in `try/except`, validate required columns with an explicit check after loading, and use `pandas` schema validation (or `pydantic` models at the FastAPI boundary) to reject malformed input early. |

---

### ISSUE-06 · No modularisation — all logic is inline

| Attribute | Detail |
|---|---|
| **Severity** | 🟠 HIGH |
| **Cells** | All code cells (2, 4, 6, 8, 10, 12) |
| **Description** | Every step — directory listing, data loading, feature engineering, training, prediction, and output — is written as top-level imperative code with no functions or classes. There is no way to call any individual step in isolation, which makes unit testing, API integration, and partial re-runs impossible. |
| **Fix** | Extract at minimum: `load_data(path)`, `engineer_features(df)`, `train_model(X, y)`, `predict(model, X)`, and `save_submission(predictions, path)` as standalone functions before wrapping in a FastAPI router. |

---

### ISSUE-07 · Missing dependency declarations

| Attribute | Detail |
|---|---|
| **Severity** | 🟠 HIGH |
| **Cells** | Cell 2 (implicit) |
| **Description** | The notebook imports `numpy`, `pandas`, and `sklearn` but there is no `requirements.txt`, `Pipfile`, `pyproject.toml`, or `environment.yml` in the repository. Reproducible builds — essential for containerised FastAPI deployment — are impossible without pinned versions. |
| **Fix** | Add a `requirements.txt` (or `pyproject.toml`) that pins at minimum `numpy`, `pandas`, `scikit-learn`, `fastapi`, `uvicorn`, and `joblib` with exact versions tested against the notebook output. |

---

### ISSUE-08 · Reproducibility gap — random seed set only on the model, not globally

| Attribute | Detail |
|---|---|
| **Severity** | 🟠 HIGH |
| **Cell** | Cell 12 |
| **Description** | `random_state=1` is passed to `RandomForestClassifier`, which is good. However, `numpy.random.seed` and Python's built-in `random.seed` are never set. Any future code that uses those global RNG streams (e.g. sampling, shuffling for a train/test split) will produce non-deterministic results across runs, making debugging and regression-testing unreliable. |
| **Fix** | Add `np.random.seed(42); random.seed(42)` at the top of the training script and document the seed value in config. |

---

### ISSUE-09 · Kaggle environment probe is dead code in any non-Kaggle runtime

| Attribute | Detail |
|---|---|
| **Severity** | 🟡 MEDIUM |
| **Cell** | Cell 2 |
| **Description** | The `os.walk('/kaggle/input')` loop is pure Kaggle scaffolding. It will silently produce no output (or raise a `FileNotFoundError` on stricter systems) outside the Kaggle platform and has no value in a production pipeline. It also imports `os` as a side-effect without declaring it as a named dependency. |
| **Fix** | Remove the cell entirely; replace with a config-driven data discovery step. |

---

### ISSUE-10 · Duplicate, non-reusable exploratory statistics cells

| Attribute | Detail |
|---|---|
| **Severity** | 🟡 MEDIUM |
| **Cells** | Cells 8 and 10 |
| **Description** | Cells 8 and 10 are structurally identical: filter `train_data` by a `Sex` value, compute `sum/len`, and print. This duplicated pattern would need to be repeated for every demographic cut. Neither result is stored in a variable for downstream use. |
| **Fix** | Consolidate into a single `survival_rate(df, group_col, group_val)` helper function and call it for each group; store results in a dict for optional API exposure as a `/statistics` endpoint. |

---

### ISSUE-11 · Output written to the current working directory without path validation

| Attribute | Detail |
|---|---|
| **Severity** | 🟡 MEDIUM |
| **Cell** | Cell 12, line 11 |
| **Description** | `output.to_csv('submission.csv', index=False)` writes to whatever directory the process was launched from. In a containerised service the working directory may be read-only, may overwrite an existing file silently, or may not be the intended output location. |
| **Fix** | Parameterise the output path via an environment variable or function argument and create the parent directory with `pathlib.Path.mkdir(parents=True, exist_ok=True)` before writing. |

---

### ISSUE-12 · `pd.get_dummies` column-alignment fragility at single-row inference time

| Attribute | Detail |
|---|---|
| **Severity** | 🟡 MEDIUM |
| **Cell** | Cell 12 |
| **Description** | Beyond the leakage concern (ISSUE-04), `pd.get_dummies` on a single-passenger DataFrame at API request time will only produce one dummy column per feature (since only one value is present), causing a shape mismatch with the trained model's expected input. This is a known production failure mode for any `get_dummies`-based encoding strategy. |
| **Fix** | Use a fitted `OneHotEncoder` with `handle_unknown='ignore'` and `drop='first'` inside a `sklearn.pipeline.Pipeline`; serialise the entire pipeline (encoder + model) so that inference is a single `.predict(row)` call. |

---

## Summary Table

| # | Issue | Severity | Cell(s) |
|---|---|---|---|
| 01 | Hardcoded Kaggle-platform absolute paths | 🔴 CRITICAL | 2, 4, 6 |
| 02 | No train/validation split or quality metric | 🔴 CRITICAL | 12 |
| 03 | Model never serialised to disk | 🔴 CRITICAL | 12 |
| 04 | Data leakage via independent `get_dummies` calls | 🔴 CRITICAL | 12 |
| 05 | No error handling anywhere | 🔴 CRITICAL | 4, 6, 8, 10, 12 |
| 06 | No modularisation — all logic inline | 🟠 HIGH | All code cells |
| 07 | Missing dependency declarations | 🟠 HIGH | — |
| 08 | Partial reproducibility (no global RNG seed) | 🟠 HIGH | 12 |
| 09 | Kaggle environment probe is dead code | 🟡 MEDIUM | 2 |
| 10 | Duplicate exploratory stats cells | 🟡 MEDIUM | 8, 10 |
| 11 | Output path not parameterised or validated | 🟡 MEDIUM | 12 |
| 12 | `get_dummies` fragility at single-row inference | 🟡 MEDIUM | 12 |

**Critical blockers: 5 · High: 3 · Medium: 4**  
This notebook cannot be dropped into a FastAPI service without resolving at minimum all five CRITICAL issues.

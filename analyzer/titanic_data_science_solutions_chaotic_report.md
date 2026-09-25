# Notebook Analysis Report
**File:** `sample_notebooks/titanic_data_science_solutions_chaotic.ipynb`  
**Purpose:** Pre-modernization audit for production FastAPI refactor  
**Total cells:** 101 (41 code, 60 markdown)

---

## 1. PLAIN-ENGLISH EXPLANATION

### What the notebook does, end to end

The notebook solves the Kaggle Titanic survival-prediction problem as a teaching
example. It proceeds in five logical phases:

#### Phase 1 — Data Loading (Cell 04)
Two CSV files are read into Pandas DataFrames: a labelled training set (`train.csv`,
891 rows) and an unlabelled test set (`test.csv`, 418 rows). A convenience list
`combine = [train_df, test_df]` is created so the same transformations can be
applied to both frames in a single loop.

#### Phase 2 — Exploratory Data Analysis / EDA (Cells 06–30)
The notebook profiles both datasets:
- Column names, dtypes, and null counts (`info()`, `describe()`)
- Pivot tables correlating categorical features (Pclass, Sex, SibSp, Parch) with
  survival rate
- Seaborn/matplotlib histograms and point plots visualising Age, Fare, Embarked,
  and Pclass against the Survived label

This phase is exploratory only; no state changes to the data occur here.

#### Phase 3 — Feature Engineering & Cleaning (Cells 32–75)
This is the bulk of the logic and the section most critical to preserve:

| Step | What happens | Cells |
|---|---|---|
| Drop irrelevant columns | `Ticket` and `Cabin` dropped from both frames | 32 |
| Title extraction | Regex extracts salutation from `Name`; rare titles collapsed to `"Rare"` | 34–38 |
| Title encoding | Title string → ordinal integer (Mr=1 … Rare=5); 0 for unknowns | 38 |
| Drop Name & PassengerId | Removed from train; PassengerId kept in test for submission | 40 |
| Sex encoding | `female`→1, `male`→0 | 42 |
| Age imputation | Missing Age filled with median Age per (Sex × Pclass) group | 48 |
| Age binning | Continuous Age → 5-band ordinal (0–4) | 50–52 |
| FamilySize creation | `SibSp + Parch + 1` | 56 |
| IsAlone flag | 1 if FamilySize==1, else 0 | 58 |
| Drop SibSp / Parch / FamilySize | Only IsAlone retained | 60 |
| Age×Class interaction term | `Age * Pclass` as new feature | 62 |
| Embarked imputation | Two missing rows filled with mode ("S") | 64–65 |
| Embarked encoding | S=0, C=1, Q=2 | 67 |
| Fare imputation | One missing test-set row filled with test-set median | 69 |
| Fare binning | Continuous Fare → 4-band ordinal (0–3) using fixed thresholds | 73 |

#### Phase 4 — Modelling (Cells 77–95)
`X_train` / `Y_train` / `X_test` are sliced from the transformed frames.
Nine classifiers are trained on the full `X_train` and scored against that same
training set (i.e. training accuracy, not held-out accuracy):

Logistic Regression, SVM (RBF), k-NN (k=3), Gaussian Naïve Bayes, Perceptron,
Linear SVC, SGD Classifier, Decision Tree, Random Forest (100 trees).

#### Phase 5 — Output (Cells 97–98)
Model accuracies are ranked in a DataFrame. The final `Y_pred` (from whichever
model ran last — Random Forest) is assembled with `PassengerId` into a submission
DataFrame. The `to_csv` call is **commented out**, so no file is written.

---

### Core logic that must survive a refactor

The following transformations are the irreducible pipeline that must be replicated
faithfully in any production service:

1. **Title extraction** — regex `' ([A-Za-z]+)\.'` on the `Name` field
2. **Title normalisation** — map rare/foreign titles to `"Rare"`, then to integer
3. **Sex encoding** — binary integer map
4. **Age imputation** — median per (Sex × Pclass) group  
   ⚠️ The group medians must be computed from **training data only** and stored,
   then applied to inference inputs
5. **Age binning** — fixed band edges `[0, 16, 32, 48, 64]`
6. **IsAlone flag** — derived from SibSp + Parch
7. **Age×Class interaction term**
8. **Embarked imputation** — mode from training data (always `"S"` for this dataset)
9. **Embarked encoding** — S=0, C=1, Q=2
10. **Fare imputation** — test-set median (must be training-set median in production)
11. **Fare binning** — fixed thresholds `[7.91, 14.454, 31]`
12. **Column order** — final feature set must match the order the model was trained on
13. **Model selection** — Random Forest with `n_estimators=100` is the final model
    used for the submission output

---

## 2. ISSUES LIST

### Issue 01 — Hardcoded Kaggle-relative file paths
| Field | Detail |
|---|---|
| **Severity** | CRITICAL |
| **Location** | Cell 04 |
| **Code** | `pd.read_csv('../input/train.csv')` / `pd.read_csv('../input/test.csv')` |
| **Description** | Paths are hard-wired to the Kaggle kernel filesystem layout (`../input/`). The files will not be found when run locally or in any other environment; the correct local paths are `data/titanic_train.csv` and `data/titanic_test.csv`. |
| **Fix** | Replace with configurable paths via environment variable or a config object (e.g., `settings.train_path`); for local development use `data/titanic_train.csv` and `data/titanic_test.csv`. |

---

### Issue 02 — Age imputation fit on combined train + test data (data leakage)
| Field | Detail |
|---|---|
| **Severity** | CRITICAL |
| **Location** | Cell 48 |
| **Code** | `for dataset in combine:` iterates over `[train_df, test_df]` when computing and filling Age medians |
| **Description** | The median Age per (Sex × Pclass) group is computed separately for each frame, which is technically not a single aggregation, but test-set rows still influence their own imputation values. More critically, `guess_ages` is overwritten during the test-set loop iteration, creating a subtle statefulness bug. In production the imputation lookup table must be frozen from training data only and stored as an artifact. |
| **Fix** | Compute `guess_ages` once from `train_df` only, then apply the same frozen lookup to both `train_df` and `test_df` separately. |

---

### Issue 03 — Embarked mode imputation uses combined list implicitly
| Field | Detail |
|---|---|
| **Severity** | HIGH |
| **Location** | Cell 64–65 |
| **Code** | `freq_port = train_df.Embarked.dropna().mode()[0]` then `for dataset in combine: dataset['Embarked'].fillna(freq_port)` |
| **Description** | The mode is correctly computed from `train_df` only, but this is coincidental and fragile. In production the imputed value must be a stored training-time constant, not recomputed at inference time. |
| **Fix** | Store `freq_port` as a pipeline parameter/artifact at training time and load it at inference time instead of recomputing it. |

---

### Issue 04 — Fare imputation uses test-set median, not training-set median
| Field | Detail |
|---|---|
| **Severity** | HIGH |
| **Location** | Cell 69 |
| **Code** | `test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)` |
| **Description** | The missing Fare value in the test set is filled using the **test-set** median, which is unavailable for a single inference request at runtime and is itself a form of data leakage. |
| **Fix** | Compute and store `fare_median` from `train_df` at training time; use that constant to fill missing Fare values in both train and test (and at inference). |

---

### Issue 05 — No random seeds on any stochastic model or operation
| Field | Detail |
|---|---|
| **Severity** | HIGH |
| **Location** | Cells 89, 91, 93, 95 (Perceptron, SGDClassifier, DecisionTreeClassifier, RandomForestClassifier) |
| **Code** | e.g., `RandomForestClassifier(n_estimators=100)` — no `random_state` argument |
| **Description** | Every stochastic classifier is instantiated without a `random_state`. Results will differ across runs, making benchmarking, debugging, and production model pinning unreliable. The `import random as rnd` at the top is imported but `rnd` is never seeded and the commented-out random age-guess code was replaced with a deterministic median, so the import is dead code. |
| **Fix** | Add `random_state=42` (or a config constant) to every classifier and to any `numpy` random operations; remove the unused `import random as rnd`. |

---

### Issue 06 — Model accuracy evaluated only on training data (no validation split)
| Field | Detail |
|---|---|
| **Severity** | CRITICAL |
| **Location** | Cells 79, 83, 85, 87, 89, 90, 91, 93, 95 |
| **Code** | e.g., `acc_log = round(logreg.score(X_train, Y_train) * 100, 2)` |
| **Description** | Every model's reported accuracy is its **training set accuracy**, not a held-out or cross-validated score. This makes all accuracy numbers optimistically inflated and meaningless for model selection. The notebook has no `train_test_split`, no cross-validation, and no held-out validation set. |
| **Fix** | Split `X_train`/`Y_train` into a training and validation subset using `train_test_split(random_state=42)` before fitting, or use `cross_val_score`; report validation accuracy, not training accuracy. |

---

### Issue 07 — All preprocessing and modelling logic is inline with no functions or classes
| Field | Detail |
|---|---|
| **Severity** | HIGH |
| **Location** | All code cells (entire notebook) |
| **Description** | Every transformation step is a top-level imperative statement relying on global mutable state (`train_df`, `test_df`, `combine`). There are no functions, no classes, and no sklearn Pipeline objects. This means the preprocessing logic cannot be called on a single inference row, cannot be unit-tested, and cannot be serialised as a pipeline artifact. |
| **Fix** | Wrap each logical phase (imputation, encoding, feature engineering) in a function or an `sklearn.base.TransformerMixin`; compose them into an `sklearn.pipeline.Pipeline` so the full preprocessing + model can be serialised with `joblib.dump`. |

---

### Issue 08 — Duplicate inline pattern for applying transformations to both frames
| Field | Detail |
|---|---|
| **Severity** | MEDIUM |
| **Location** | Cells 34, 36, 38, 42, 48, 52, 56, 58, 62, 65, 67, 73 |
| **Code** | `for dataset in combine:` repeated ~12 times with the same loop structure |
| **Description** | The `for dataset in combine` pattern is used identically across a dozen cells to apply transformations to both train and test. This is a repeated pattern that should be a function. One misplaced cell or a rerun of a subset of cells corrupts the shared mutable state irreversibly. |
| **Fix** | Extract each transformation into a pure function that accepts a DataFrame and returns a transformed DataFrame; call the function explicitly on `train_df` and `test_df`. |

---

### Issue 09 — Silent data corruption bug in Age binning (Cell 52)
| Field | Detail |
|---|---|
| **Severity** | HIGH |
| **Location** | Cell 52 |
| **Code** | `dataset.loc[ dataset['Age'] > 64, 'Age']` — missing `= 4` assignment |
| **Description** | The last Age band (age > 64 → ordinal 4) is missing the assignment value. The expression is a no-op read instead of a write. Ages above 64 remain as the raw imputed integer rather than being binned to 4, silently producing incorrect feature values for elderly passengers. |
| **Fix** | Change the last line to `dataset.loc[dataset['Age'] > 64, 'Age'] = 4`. |

---

### Issue 10 — No error handling anywhere in the notebook
| Field | Detail |
|---|---|
| **Severity** | HIGH |
| **Location** | Cells 04, 42, 67 (and throughout) |
| **Description** | No `try/except` blocks exist. If the CSV files are missing the notebook crashes with an unhandled `FileNotFoundError`. If a new passenger record arrives with an unseen `Embarked` value the `.map({'S':0,'C':1,'Q':2}).astype(int)` call will raise a `ValueError` on the `.astype(int)` because `map` returns `NaN` for unmapped keys. Same risk for Sex and Title encoding. |
| **Fix** | Add file-existence checks before `read_csv`; use `.map(...).fillna(-1).astype(int)` or explicit unknown-category handling for all categorical maps; wrap the inference path in a `try/except` with structured error responses. |

---

### Issue 11 — Submission output path is commented out; no artefact is saved
| Field | Detail |
|---|---|
| **Severity** | MEDIUM |
| **Location** | Cell 98 |
| **Code** | `# submission.to_csv('../output/submission.csv', index=False)` |
| **Description** | The only output-producing line in the notebook is commented out. The trained model is also never serialised (`joblib.dump` is absent), so re-running the notebook is the only way to regenerate predictions. There is no saved model artefact. |
| **Fix** | Uncomment and redirect to `data/submission.csv` (or a configurable output path); add `joblib.dump(random_forest, 'models/titanic_rf.joblib')` to persist the trained model. |

---

### Issue 12 — No requirements file or environment specification
| Field | Detail |
|---|---|
| **Severity** | HIGH |
| **Location** | Repository root (absent) |
| **Description** | There is no `requirements.txt`, `pyproject.toml`, `environment.yml`, or `Pipfile`. The notebook depends on `pandas`, `numpy`, `scikit-learn`, `seaborn`, and `matplotlib`, but their versions are completely undeclared. `sklearn` API surface (e.g., `FacetGrid(size=…)` vs `height=…`) is version-sensitive and will silently break on mismatched installs. |
| **Fix** | Create a `requirements.txt` (or `pyproject.toml`) pinning at minimum: `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`, and `joblib` with version constraints. |

---

### Issue 13 — `size` parameter in `FacetGrid` is deprecated
| Field | Detail |
|---|---|
| **Severity** | MEDIUM |
| **Location** | Cells 26, 28, 30, 44 |
| **Code** | `sns.FacetGrid(..., size=2.2, ...)` |
| **Description** | The `size` keyword was renamed to `height` in seaborn 0.9.0 (2018). Any environment running seaborn ≥ 0.9 will emit a `FutureWarning`; seaborn ≥ 0.13 raises a `TypeError`. |
| **Fix** | Replace `size=` with `height=` in all `FacetGrid` calls. |

---

### Issue 14 — `"After"` string expression in Cell 32 is a dead no-op
| Field | Detail |
|---|---|
| **Severity** | MEDIUM |
| **Location** | Cell 32 |
| **Code** | `"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape` — missing `print(...)` wrapper |
| **Description** | The intended debug print is a bare tuple expression. In a `.py` script it is silently discarded; in the notebook it only displays because Jupyter renders the last expression of a cell. This pattern is misleading and will not behave as expected in a `.py` service. |
| **Fix** | Wrap in `print(...)` or remove it; in production code replace with structured logging. |

---

### Issue 15 — `import random as rnd` is dead code
| Field | Detail |
|---|---|
| **Severity** | MEDIUM |
| **Location** | Cell 02 |
| **Code** | `import random as rnd` |
| **Description** | The only use of `rnd` was `rnd.uniform(...)` in the Age imputation, which is commented out. The import is never called anywhere in the executed code path. |
| **Fix** | Remove the import; if randomness is re-introduced, seed it explicitly with `random.seed(42)`. |

---

### Issue 16 — All models are trained and `Y_pred` is silently overwritten on each cell run
| Field | Detail |
|---|---|
| **Severity** | HIGH |
| **Location** | Cells 79, 83, 85, 87, 89, 90, 91, 93, 95 |
| **Code** | Every model block reassigns `Y_pred = model.predict(X_test)` |
| **Description** | `Y_pred` is a single global variable that is overwritten by each of the nine model cells. The submission cell (98) uses whichever model ran last. If cells are run out of order (easy in a notebook) the submission silently uses the wrong model's predictions. |
| **Fix** | Use model-specific variable names (e.g., `y_pred_rf`) and explicitly reference the chosen model's predictions when building the submission. |

---

## Summary Table

| # | Issue | Severity | Cell(s) |
|---|---|---|---|
| 01 | Hardcoded Kaggle `../input/` paths | **CRITICAL** | 04 |
| 02 | Age imputation leaks test-set data | **CRITICAL** | 48 |
| 03 | Embarked imputation not stored as training artifact | **HIGH** | 64–65 |
| 04 | Fare imputation uses test-set median | **HIGH** | 69 |
| 05 | No random seeds on stochastic models | **HIGH** | 89, 91, 93, 95 |
| 06 | Accuracy scored on training data only — no validation split | **CRITICAL** | 79–95 |
| 07 | No modularisation — all logic is inline global state | **HIGH** | All code cells |
| 08 | Repeated `for dataset in combine` pattern — no functions | **MEDIUM** | 34–73 |
| 09 | Silent bug: Age > 64 bin assignment missing `= 4` | **HIGH** | 52 |
| 10 | No error handling for file I/O or encoding mismatches | **HIGH** | 04, 42, 67 |
| 11 | Model not serialised; submission output commented out | **MEDIUM** | 98 |
| 12 | No requirements.txt or environment specification | **HIGH** | (root) |
| 13 | Deprecated `size=` parameter in FacetGrid | **MEDIUM** | 26, 28, 30, 44 |
| 14 | "After" debug line is a bare expression, not a print | **MEDIUM** | 32 |
| 15 | `import random as rnd` is dead code | **MEDIUM** | 02 |
| 16 | `Y_pred` overwritten silently across all model cells | **HIGH** | 79–95 |

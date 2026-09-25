# Notebook Analysis Report
## `data_science_framework_99acc_heavy.ipynb`

---

## 1. Plain-English Explanation

### What the notebook does, end to end

This is a Kaggle tutorial notebook for the classic **Titanic survival prediction** competition. It walks through a full data science workflow across 58 cells (35 code, rest markdown). Here is what each phase does:

#### Phase 1 — Environment Setup (Cells 5, 7)
Imports standard data science libraries (`pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`) and configures plotting defaults. Also calls a Kaggle-only shell command to list files in `../input`.

#### Phase 2 — Data Loading (Cell 9)
Reads two CSV files from hardcoded Kaggle paths:
- `../input/train.csv` → stored as `data_raw` (labeled training data, 891 rows)
- `../input/test.csv` → stored as `data_val` (competition hold-out / validation, 418 rows)

Both are added to a shared list `data_cleaner = [data1, data_val]` so cleaning loops apply to both simultaneously.

#### Phase 3 — Data Cleaning (Cells 11, 13)
- **Null inspection**: Counts missing values in both datasets.
- **Completing**: Fills missing `Age` with per-dataset median, `Embarked` with per-dataset mode, `Fare` with per-dataset median.
- **Dropping**: Removes `PassengerId`, `Cabin`, and `Ticket` from the train copy (`data1`) only — `PassengerId` is retained in `data_val` for the submission file.

#### Phase 4 — Feature Engineering (Cell 14)
Applied to both `data1` and `data_val` inside a single loop:
- `FamilySize` = `SibSp + Parch + 1`
- `IsAlone` = 1 if `FamilySize == 1`, else 0
- `Title` = extracted from the passenger's `Name` string (e.g. "Mr", "Miss", "Mrs"); rare titles (< 10 occurrences) are bucketed as `"Misc"` — but the rarity threshold is computed **only from the training data** and then applied to both datasets.
- `FareBin` = quartile bin via `pd.qcut(Fare, 4)`
- `AgeBin` = equal-width bin via `pd.cut(Age, 5)`

#### Phase 5 — Encoding / Format Conversion (Cell 16)
- `LabelEncoder` applied to `Sex`, `Embarked`, `Title`, `AgeBin`, `FareBin` for both datasets.
- Three feature-set variants are defined for modeling: raw numeric (`data1_x_calc`), binned (`data1_x_bin`), and one-hot dummy (`data1_dummy`).

#### Phase 6 — Train / Test Split (Cell 20)
`train_test_split` with `random_state=0` (75/25 default split) is applied to `data1` three times — once per feature set variant. The resulting split objects (`train1_x`, `test1_x`, etc.) are **never used in the actual model training** that follows; the full `data1` is used instead.

#### Phase 7 — EDA / Visualization (Cells 22–32)
Extensive exploratory plots: survival rates grouped by discrete features, boxplots, violin plots, histograms, pair plots, seaborn FacetGrids, and a Pearson correlation heatmap. Purely analytical — no state that carries into production.

#### Phase 8 — Bulk Model Benchmarking (Cell 34)
Initializes 22 classifiers (AdaBoost, Bagging, ExtraTrees, GradientBoosting, RandomForest, GaussianProcess, LogisticRegressionCV, PassiveAggressive, RidgeClassifier, SGD, Perceptron, BernoulliNB, GaussianNB, KNN, SVC, NuSVC, LinearSVC, DecisionTree, ExtraTree, LDA, QDA, XGBoost).

Each model is:
1. Cross-validated with `ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.6, random_state=0)` on `data1[data1_x_bin]`.
2. Immediately re-fit on the **full** `data1[data1_x_bin]` — training data contamination.
3. Used to predict on `data1[data1_x_bin]` (training set) and the predictions are stored in `MLA_predict`.

#### Phase 9 — Handmade / Baseline Models (Cells 37–40)
- A **coin-flip baseline** that randomly assigns 0/1 without a seed.
- A **hand-coded decision tree** (`mytree()`) that encodes survival rules directly in Python if-else logic (female survives, except 3rd class + Southampton + Fare > 8; master males survive).

#### Phase 10 — Decision Tree Hyperparameter Tuning (Cell 43)
`GridSearchCV` over `criterion` × `max_depth` on `data1[data1_x_bin]`, scoring by ROC-AUC. Best model is retained in `tune_model`.

#### Phase 11 — Recursive Feature Elimination (Cell 45)
`RFECV` applied to `dtree` on `data1[data1_x_bin]` to select an optimal feature subset `X_rfe`. The selected subset is cross-validated and then grid-searched again.

#### Phase 12 — Ensemble Voting (Cells 49, 51–52)
- **Untuned voting**: Hard and soft `VotingClassifier` over 12 selected estimators (those with `predict_proba`), cross-validated then fit on full `data1`.
- **Tuned voting** (Cell 51): Individually grid-searches each estimator with `GridSearchCV`, then updates them in-place. This cell is computationally intensive (self-described as ~5.5 minutes) and explicitly marked "not production ready."
- **Final ensemble** (Cell 52): Re-builds `grid_hard` and `grid_soft` with the tuned estimators and fits on full `data1`.

#### Phase 13 — Submission Generation (Cell 53)
Applies the **tuned hard voting classifier** (`grid_hard`) to `data_val[data1_x_bin]` to produce predictions. Also applies `mytree()` (hand-coded model) but then overwrites it with `grid_hard`. Writes `../working/submit.csv` with `PassengerId` and `Survived` columns.

---

### Core logic that must survive a refactor

For a production FastAPI service, the following pipeline logic is the irreducible core:

1. **Imputation** — fill `Age` (median), `Embarked` (mode), `Fare` (median) using statistics derived **only from training data**, then applied to inference inputs.
2. **Feature engineering** — compute `FamilySize`, `IsAlone`, extract `Title` from `Name`, apply `FareBin` (qcut) and `AgeBin` (cut) using **training-fitted bin edges**, bucket rare titles as `"Misc"` using a **training-derived threshold**.
3. **Encoding** — `LabelEncoder` (or `OrdinalEncoder`) for `Sex`, `Embarked`, `Title`, `AgeBin`, `FareBin` fitted on training data.
4. **Feature set** — `data1_x_bin = ['Sex_Code', 'Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']`
5. **Model** — tuned hard `VotingClassifier` (`grid_hard`) over the 12 estimators with tuned hyperparameters.
6. **Prediction output** — binary integer `Survived` ∈ {0, 1} per passenger.

The EDA cells, visualizations, coin-flip baseline, handmade tree, and the "under construction" grid search (Cell 50) are all non-essential and should be dropped.

---

## 2. Issues List

---

### ISSUE-01 — Hardcoded Kaggle-Specific File Paths
**Severity: CRITICAL**

| Location | Code |
|---|---|
| Cell 5 (code cell 0) | `check_output(["ls", "../input"])` — Unix shell call, Windows-incompatible |
| Cell 9 (code cell 2) | `pd.read_csv('../input/train.csv')` |
| Cell 9 (code cell 2) | `pd.read_csv('../input/test.csv')` |
| Cell 53 (code cell 34) | `submit.to_csv("../working/submit.csv", ...)` |

The paths `../input/` and `../working/` are Kaggle kernel sandbox paths that do not exist on any other machine or in any deployment environment.

**Fix:** Replace with configurable paths (e.g. environment variables or a config object); the data already exists at `data/titanic_train.csv` and `data/titanic_test.csv` at the project root.

---

### ISSUE-02 — Data Leakage: Label Encoding Fit on Both Train and Validation Simultaneously
**Severity: CRITICAL**

| Location | Code |
|---|---|
| Cell 16 (code cell 6) | `for dataset in data_cleaner: dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])` |

`LabelEncoder.fit_transform()` is called inside a loop over `[data1, data_val]`. This means the encoder is **re-fitted on `data_val`** (the hold-out/test set), allowing label categories present only in `data_val` to influence the encoding. Correct practice is to `fit` on training data and `transform` on everything else.

**Fix:** Call `label.fit(data1[col])` once on the training set, then call `.transform()` on both datasets separately.

---

### ISSUE-03 — Data Leakage: Missing Value Imputation Uses Per-Dataset Statistics
**Severity: HIGH**

| Location | Code |
|---|---|
| Cell 13 (code cell 4) | `dataset['Age'].fillna(dataset['Age'].median(), inplace=True)` (inside `for dataset in data_cleaner`) |
| Cell 13 (code cell 4) | `dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)` |
| Cell 13 (code cell 4) | `dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)` |

Each dataset (including `data_val`) computes its own median/mode for imputation. In production, only training-set statistics should be used and persisted. This leaks the test-set distribution into imputed values.

**Fix:** Compute median and mode on `data1` (training set) only, store those values, and apply them to both datasets.

---

### ISSUE-04 — Data Leakage: `FareBin` and `AgeBin` Bin Edges Not Persisted from Training
**Severity: HIGH**

| Location | Code |
|---|---|
| Cell 14 (code cell 5) | `dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)` (inside `for dataset in data_cleaner`) |
| Cell 14 (code cell 5) | `dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)` |

`pd.qcut` and `pd.cut` are called independently on each dataset. `qcut` computes quantile boundaries from each dataset's own distribution. In production the bin edges must be computed on training data and reused for all subsequent inputs.

**Fix:** Capture bin edges from `pd.qcut`/`pd.cut` on `data1`, then use `pd.cut(..., bins=saved_edges)` on `data_val` and incoming inference data.

---

### ISSUE-05 — Models Trained and Evaluated on the Same Full Dataset (No Train/Test Separation in Practice)
**Severity: CRITICAL**

| Location | Code |
|---|---|
| Cell 34 (code cell 20) | `alg.fit(data1[data1_x_bin], data1[Target])` then `MLA_predict[MLA_name] = alg.predict(data1[data1_x_bin])` |
| Cell 49 (code cell 30) | `vote_hard.fit(data1[data1_x_bin], data1[Target])` |
| Cell 52 (code cell 33) | `grid_hard.fit(data1[data1_x_bin], data1[Target])` |
| Cell 43 (code cell 26) | `dtree.fit(data1[data1_x_bin], data1[Target])` — then predictions reused from base |

Although `train_test_split` is called in Cell 20, those split variables (`train1_x`, `test1_x`, etc.) are **never used** for any actual model fitting. All models are fit on the full `data1`. Cross-validation via `ShuffleSplit` partially mitigates this during evaluation, but the final fitted models are overfit to the full training set.

**Fix:** Train the final model on `train1_x_bin` only; evaluate generalization on `test1_x_bin`; never fit on `data_val`.

---

### ISSUE-06 — No Random Seed for the Coin-Flip Baseline Model
**Severity: MEDIUM**

| Location | Code |
|---|---|
| Cell 37 (code cell 22) | `if random.random() > .5:` — no `random.seed()` call |

The random baseline produces a different result on every run, making it non-reproducible as a comparison baseline.

**Fix:** Add `random.seed(0)` (or any fixed integer) before the loop, consistent with the `random_state=0` used elsewhere in the notebook.

---

### ISSUE-07 — No Modularization (All Logic is Inline)
**Severity: HIGH**

| Location | Code |
|---|---|
| Cells 13, 14, 16 | Cleaning, feature engineering, and encoding are all inline for-loops with no wrapping function |
| Cell 53 (code cell 34) | Prediction and submission generation is a flat script block |

No reusable functions or classes exist for the preprocessing pipeline (except the isolated `mytree()` and `correlation_heatmap()` utilities). This means the **exact same data transformation logic must be reproduced manually** every time a new input needs to be scored, making a FastAPI service impossible without a rewrite.

**Fix:** Extract `clean_data()`, `engineer_features()`, `encode_features()`, and `predict()` as standalone functions or a `sklearn.pipeline.Pipeline`; use `joblib` to serialize the fitted pipeline.

---

### ISSUE-08 — Deprecated and Broken API Calls
**Severity: HIGH**

| Location | Code |
|---|---|
| Cell 7 (code cell 1) | `from pandas.tools.plotting import scatter_matrix` — removed in pandas ≥ 0.25; correct import is `from pandas.plotting import scatter_matrix` |
| Cell 14 (code cell 5) | `dataset['IsAlone'].loc[...]` chained assignment — raises `SettingWithCopyWarning` and is unreliable |
| Cell 37 (code cell 22) | `data1.set_value(index, 'Random_Predict', 1)` — `set_value` was removed in pandas 0.25; use `.at[]` or `.loc[]` |
| Cell 5 (code cell 0) | `check_output(["ls", "../input"])` — `ls` is a Unix command; crashes on Windows |
| Cell 31 (code cell 18) | `sns.pairplot(..., size=1.2)` — `size` parameter renamed to `height` in seaborn ≥ 0.9 |

**Fix:** Update imports and API calls to match current library versions; treat these as build-blocking errors for a production service.

---

### ISSUE-09 — Missing Dependency Declaration
**Severity: HIGH**

| Location | Code |
|---|---|
| Cell 5 (code cell 0) | Comment says "defined by the kaggle/python docker image" — relies on a Kaggle-managed environment |
| Repo root | No `requirements.txt`, `environment.yml`, `pyproject.toml`, or `setup.cfg` present |

There is no pinned specification of which library versions are required. The notebook uses APIs from pandas, seaborn, and sklearn that have changed across versions, so version pinning is essential.

**Fix:** Create `requirements.txt` (or `pyproject.toml`) with pinned versions for at minimum: `pandas`, `numpy`, `scipy`, `scikit-learn`, `xgboost`, `seaborn`, `matplotlib`, `graphviz`.

---

### ISSUE-10 — No Error Handling for File I/O or Data Quality
**Severity: HIGH**

| Location | Code |
|---|---|
| Cell 9 (code cell 2) | `pd.read_csv('../input/train.csv')` — no try/except; if the file is missing the entire notebook crashes with an unguarded `FileNotFoundError` |
| Cell 13 (code cell 4) | `dataset['Embarked'].mode()[0]` — if `Embarked` is entirely null, `.mode()` returns an empty Series and `[0]` raises `IndexError` |
| Cell 14 (code cell 5) | `dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]` — if any name lacks a comma or period, this silently produces `NaN` titles without warning |
| Cell 16 (code cell 6) | `label.fit_transform(dataset['AgeBin'])` — if any bin value in `data_val` was not seen during fitting, `LabelEncoder` raises `ValueError: y contains previously unseen labels` |

**Fix:** Wrap file loading in try/except with descriptive messages; add assertions or validation checks after each cleaning step; switch to `OrdinalEncoder(handle_unknown='use_encoded_value')` to handle unseen categories gracefully.

---

### ISSUE-11 — Repeated Null-Check Diagnostic Cells
**Severity: MEDIUM**

| Location | Code |
|---|---|
| Cell 11 (code cell 3) | `data1.isnull().sum()` + `data_val.isnull().sum()` + `data_raw.describe()` |
| Cell 18 (code cell 7) | Identical null-count print block for `data1` and `data_val` + `data_raw.describe()` |

The same diagnostic output is printed twice — once before cleaning and once after — using duplicated inline code rather than a reusable function. In a service context this is dead weight.

**Fix:** Extract a `summarize_nulls(df, label)` helper called once before and once after cleaning.

---

### ISSUE-12 — `graphviz` System Dependency with No Installation Check
**Severity: MEDIUM**

| Location | Code |
|---|---|
| Cell 46 (code cell 28) | `import graphviz` then `graphviz.Source(dot_data)` |

`graphviz` requires both a Python package **and** a system-level Graphviz binary installation. On most servers / CI environments this binary is absent, causing a silent or cryptic failure. There is no `try/except` or check for the binary.

**Fix:** Wrap the graphviz cell in a try/except block, or remove it entirely since tree visualization is non-essential to the prediction pipeline.

---

### ISSUE-13 — Cell 50 is Explicitly Incomplete / Under Construction
**Severity: MEDIUM**

| Location | Code |
|---|---|
| Cell 50 (code cell 31) | Comment: `#IMPORTANT: THIS SECTION IS UNDER CONSTRUCTION!!!! 12.24.17` and all the meaningful code is commented out |

This cell defines `vote_param` (a large hyperparameter grid for a soft-vote ensemble) but the actual `GridSearchCV` call is commented out. It contributes no executable logic but would confuse any engineer inheriting the code.

**Fix:** Delete the cell entirely; the tuned grid search in Cell 51 supersedes it.

---

### ISSUE-14 — IPython/Jupyter-Only Display Dependency in Production Code
**Severity: MEDIUM**

| Location | Code |
|---|---|
| Cell 5 (code cell 0) | `from IPython import display` |
| Cell 7 (code cell 1) | `%matplotlib inline` — IPython magic command |
| Cell 46 (code cell 28) | `graph` (bare expression to render graphviz in Jupyter) |

IPython magic commands (`%matplotlib inline`) and bare expression rendering are Jupyter-only constructs that will raise `SyntaxError` or do nothing when run as a plain Python script or inside a FastAPI worker.

**Fix:** Remove all `%matplotlib` magics and bare display expressions from the production code path; configure matplotlib backend explicitly (`matplotlib.use('Agg')`) if plots are needed server-side.

---

### Summary Table

| # | Issue | Severity | Cell(s) |
|---|---|---|---|
| 01 | Hardcoded Kaggle-specific file paths | **CRITICAL** | 5, 9, 53 |
| 02 | Label encoder fit on validation set (data leakage) | **CRITICAL** | 16 |
| 03 | Imputation uses per-dataset statistics (data leakage) | **HIGH** | 13 |
| 04 | Bin edges recomputed per dataset (data leakage) | **HIGH** | 14 |
| 05 | Train/test split defined but never used; all models fit on full data | **CRITICAL** | 20, 34, 49, 52 |
| 06 | No random seed on coin-flip baseline | **MEDIUM** | 37 |
| 07 | No modularization — all logic inline | **HIGH** | 13, 14, 16, 53 |
| 08 | Deprecated / broken API calls (pandas, seaborn, subprocess) | **HIGH** | 5, 7, 14, 31, 37 |
| 09 | No dependency declaration (no requirements.txt / env spec) | **HIGH** | repo root |
| 10 | No error handling for file I/O, nulls, or unseen categories | **HIGH** | 9, 13, 14, 16 |
| 11 | Repeated duplicate diagnostic cells | **MEDIUM** | 11, 18 |
| 12 | `graphviz` system binary assumed present with no guard | **MEDIUM** | 46 |
| 13 | Cell 50 is explicitly incomplete / dead commented-out code | **MEDIUM** | 50 |
| 14 | IPython/Jupyter-only constructs in code path | **MEDIUM** | 5, 7, 46 |

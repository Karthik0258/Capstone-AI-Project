# Analytics Module — Titanic Dataset

## Overview
This module demonstrates the full analyst‑to‑data‑scientist workflow in one pass:
- Profile the Titanic dataset
- Handle imperfections defensibly
- Build a clear visual story
- Train and rigorously evaluate predictive models
- Save a deployable pipeline artifact

Dataset loaded once via `sns.load_dataset("titanic")` and saved as `titanic.csv` for offline fallback.

---

## Part A — EDA & Data Story

### Profiling & Cleaning
- Dataset shape, info, describe outputs
- Missing value percentages per column
- Strategy applied (drop, impute, encode missing)

### Univariate Analysis
- Histograms & boxplots for `age` and `fare`
- Outlier counts via IQR rule
- Skewness conclusion for `fare` (mean vs. median vs. mode)

### Bivariate Analysis
- Survival rates by sex, pclass, sex+pclass
- Correlation heatmap (survived, pclass, age, sibsp, parch, fare)
- Interpretation of top 2 strongest correlations

### Multivariate Story
- At least 4 charts (bar, scatter, pairplot, heatmap)
- Each with 2–4 sentence interpretation

### Standardization Check
- Z‑score transform of `age` and `fare`
- Before/after comparison showing mean ≈ 0, std ≈ 1

---

## Part B — Predictive Modeling

### Train/Test Split
- Stratified split justified by class imbalance

### Preprocessing
- ColumnTransformer + Pipeline
- Train‑only fit for imputation, encoding, scaling

### Models Trained
- Logistic Regression
- Decision Tree (visualized with `plot_tree`)
- Random Forest

### Evaluation
- Confusion matrix
- Accuracy, Precision, Recall, F1
- ROC curve + AUC
- Side‑by‑side comparison table

### Imbalance Handling
- Baseline vs. `class_weight="balanced"` vs. SMOTE
- Metrics compared, conclusion on best strategy

### Hyperparameter Tuning
- GridSearchCV on Random Forest (`n_estimators`, `max_depth`, `max_features`)
- Best parameters + OOB score reported

### Regression Side‑Task
- Target: `fare`
- Metrics: MAE, RMSE, R², Adjusted R²
- Residual plot with heteroscedasticity conclusion

---

## Final Comparison

### Classifier Metrics
| Model              | Accuracy | Precision | Recall | F1 | ROC AUC |
|--------------------|----------|-----------|--------|----|---------|
| Logistic Regression|          |           |        |    |         |
| Decision Tree      |          |           |        |    |         |
| Random Forest      |          |           |        |    |         |

### Regression Metrics
| Model             | MAE | RMSE | R² | Adjusted R² |
|-------------------|-----|------|----|-------------|
| Linear Regression |     |      |    |             |

---

## Recommendation
Based on the evaluation:
- **Best classifier**: [Insert model name] — chosen for its superior [metric values].
- **Reasoning**: Reference accuracy, precision/recall balance, F1, and ROC AUC.
- The regression task provided useful insights into fare prediction but is not directly comparable to classification metrics.

---

## Saved Pipeline
- Full preprocessing + estimator pipeline saved via `joblib.dump`
- Reload confirmed with `joblib.load` on raw input

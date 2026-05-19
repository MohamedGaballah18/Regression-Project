# 📊 Student Exam Score Prediction — Regression Project

A from-scratch implementation of Linear Regression applied to predicting students' final exam scores, covering three mathematical perspectives: **Statistical**, **Numerical**, and **Machine Learning (Gradient Descent)** — built without any ML libraries for the core models.

---

## 📁 Dataset

**Students_data.csv** — 4,000+ rows of student academic data.

| Feature | Description |
|---|---|
| `Previous_Scores` | Past academic performance |
| `Hours_Studied` | Weekly study hours |
| `Attendance` | Attendance rate |
| `Sleep_Hours` | Average sleep per night |
| `Motivation_Level` | Low / Medium / High |
| `Parental_Involvement` | Low / Medium / High |
| `Internet_Access` | Yes / No |
| `Extracurricular_Activities` | Yes / No |
| `Tutoring_Sessions` | Number of sessions |
| `Access_to_Resources` | Low / Medium / High |
| **`Final_Exam_Score`** | **Target variable** |

---

## ⚙️ Pipeline

```
Raw Data → EDA → Train/Test Split (80/20) → Missing Value Imputation
         → Outlier Capping (IQR) → Ordinal Encoding → Z-score Standardization
         → Correlation Analysis → Best Predictor Selection → Modeling
```

**Preprocessing details:**
- Missing values filled using train-set median (numerical) and mode (categorical)
- Outliers capped using IQR method — no rows dropped
- Ordinal encoding for hierarchical categories (Low=0, Medium=1, High=2)
- Z-score standardization fitted on train set only to prevent data leakage

---

## 🧠 Models

All models were implemented from scratch using NumPy — no Scikit-learn.

---

### 1. Statistical View — `LinearRegressionStatisticalView`

Solves the regression analytically using variance/covariance formulas and provides full statistical inference.

```python
model = LinearRegressionStatisticalView(alpha=0.05)
model.fit(X_train['Previous_Scores'].values, y_train.values)
print(model.summary())
```

**Outputs:**

| Metric | Description |
|---|---|
| `β̂₀`, `β̂₁` | Intercept and slope |
| `R²` | Coefficient of determination |
| `Pearson r` | Linear correlation strength |
| `F-Statistic` | Overall model significance |
| `F-Critical` | Rejection threshold at α = 0.05 |
| `Confidence Intervals` | 95% CI for β₀ and β₁ |
| Degrees of Freedom | df_model, df_resid, df_total |

A `compare()` helper function selects the best predictor between two features based on F-statistic.

---

### 2. Numerical View — `SimpleLinearRegressionNumericalView`

Treats regression as a system of linear equations and solves `Ax = b` directly via `np.linalg.solve`.

---

### 3. Machine Learning View — `MLGradientDescent`

Iteratively minimizes MSE using partial derivatives with a configurable learning rate and number of iterations.

---

Each model was tested with **OLS**, **Ridge (L2)**, and **Lasso (L1)** regularization across all three views.

---

## 📊 Key Formulas

**Slope and Intercept (Statistical View):**

$$B_1 = \frac{S_{XY}}{S_{XX}} = \frac{\sum x_i y_i - n\bar{x}\bar{y}}{\sum x_i^2 - n\bar{x}^2}, \quad B_0 = \bar{y} - B_1\bar{x}$$

**Normal Equation (Numerical View):**

$$\begin{bmatrix} n & \sum x_i \\ \sum x_i & \sum x_i^2 \end{bmatrix} \begin{bmatrix} B_0 \\ B_1 \end{bmatrix} = \begin{bmatrix} \sum y_i \\ \sum x_i y_i \end{bmatrix}$$

**Gradient Descent Update Rule (ML View):**

$$B_0 \leftarrow B_0 - \alpha \cdot \frac{2}{n}\sum(\hat{y}_i - y_i), \quad B_1 \leftarrow B_1 - \alpha \cdot \frac{2}{n}\sum(\hat{y}_i - y_i) \cdot x_i$$

**F-Statistic (Hypothesis Testing):**

$$F = \frac{SSR / df_{model}}{SSE / df_{resid}}$$

> If **F > F-critical** → Reject H₀ → the predictor is statistically significant.

---

## 📈 Results

### Statistical View — Best Single Predictor

`Previous_Scores` was identified as the best single predictor with R² ≈ **0.40**, and H₀ was rejected at α = 0.05.

### Model Specification Comparison (Gradient Descent)

| Specification | Predictors | R² | Adj. R² | Test MSE |
|---|:---:|:---:|:---:|:---:|
| Best Single Feature | 1 | 0.3967 | 0.3965 | 45.60 |
| Domain-Based | 4 | 0.7291 | 0.7287 | 21.64 |
| Full Model | 10 | **0.7918** | **0.7912** | **15.60** |

### Regularization Comparison (Simple Regression)

| Model | R² | Test MSE |
|---|:---:|:---:|
| Standard OLS | 0.3967 | 45.60 |
| Ridge (L2) | 0.3963 | 45.80 |
| Lasso (L1) | 0.3967 | 45.61 |

Adding more features consistently reduced error — the full model nearly doubled the explanatory power of the single-predictor model.

---

## 🛠️ Tech Stack

- **Python** — NumPy, Pandas, SciPy
- **Visualization** — Matplotlib, Seaborn, Plotly

---

## 🚀 Usage

```bash
git clone https://github.com/MohamedGaballah18/Regression-Project.git
cd Regression-Project
```

Run the full project:
```bash
jupyter notebook finalregressionproject.ipynb
```

Run the statistical view standalone:
```bash
jupyter notebook linear_regression_statistical_view.ipynb
```

> Make sure `Students_data.csv` is in the same directory as the notebooks.
git clone https://github.com/MohamedGaballah18/Regression-Project.git
cd Regression-Project
jupyter notebook finalregressionproject.ipynb
```

> Make sure `Students_data.csv` is in the same directory as the notebook.

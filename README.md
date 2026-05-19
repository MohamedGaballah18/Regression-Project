# 📊 Student Exam Score Prediction — Regression Project

A from-scratch implementation of Linear Regression applied to predicting students' final exam scores, built without using any ML libraries for the core models.

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
Raw Data → EDA → Train/Test Split → Missing Values → Outlier Capping (IQR)
         → Encoding → Standardization → Feature Selection → Modeling
```

**Preprocessing steps:**
- Missing values filled using train-set median (numerical) and mode (categorical)
- Outliers capped using IQR method (no rows dropped)
- Ordinal encoding for hierarchical categories (Low=0, Medium=1, High=2)
- Z-score standardization fitted on train set only

---

## 🧠 Models

All three models were implemented from scratch using NumPy — no Scikit-learn.

### 1. Statistical View
Solves directly using variance/covariance formulas (SXX, SXY).

### 2. Numerical View
Solves the Normal Equation as a system of linear equations `Ax = b` via `np.linalg.solve`.

### 3. Machine Learning View (Gradient Descent)
Iteratively minimizes MSE using partial derivatives with a configurable learning rate.

Each model was tested with **OLS**, **Ridge (L2)**, and **Lasso (L1)** regularization.

---

## 📈 Results

| Specification | Predictors | R² | Test MSE |
|---|:---:|:---:|:---:|
| Best Single Feature | 1 | 0.3967 | 45.60 |
| Domain-Based | 4 | 0.7291 | 21.64 |
| Full Model | 10 | **0.7918** | **15.60** |

The full model (10 features) achieved the best performance with **R² = 0.79**, and adding more features consistently reduced test error without signs of overfitting.

---

## 🔢 Math Behind the Models

**Statistical View — Slope & Intercept:**

$$B_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2}, \quad B_0 = \bar{y} - B_1\bar{x}$$

**Gradient Descent — Update Rule:**

$$B_0 \leftarrow B_0 - \alpha \cdot \frac{2}{n}\sum(\hat{y}_i - y_i)$$
$$B_1 \leftarrow B_1 - \alpha \cdot \frac{2}{n}\sum(\hat{y}_i - y_i) \cdot x_i$$

---

## 🛠️ Tech Stack

- **Python** — NumPy, Pandas
- **Visualization** — Matplotlib, Seaborn, Plotly

---

## 🚀 Usage

```bash
git clone https://github.com/MohamedGaballah18/Regression-Project.git
cd Regression-Project
jupyter notebook finalregressionproject.ipynb
```

> Make sure `Students_data.csv` is in the same directory as the notebook.

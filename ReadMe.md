# 🩺 Diabetes Prediction with Machine Learning

This project uses the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) to predict the likelihood of diabetes in female patients based on several medical parameters.

## 📊 Dataset Overview

The dataset contains 768 entries and 9 columns:

- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`
- `Outcome` (Target variable: 1 = diabetic, 0 = non-diabetic)

## 🧹 Data Preprocessing

- Replaced invalid `0` values with `NaN` in key columns like `Glucose`, `BloodPressure`, `Insulin`, etc.
- Filled missing values with the **mean** of each column.
- Scaled features using **StandardScaler**.
- Engineered additional features:
  - `AgeGroup` (binned age)
  - `BMI_Glucose_Ratio` (BMI divided by Glucose)

## 🤖 Model Training

Trained multiple models and evaluated their performance:

- **Logistic Regression** – Baseline accuracy ~75.3%
- **Random Forest Classifier** – Tuned via `GridSearchCV`  
  - ✅ Best cross-validation accuracy: **77.36%**  
  - ✅ Final test accuracy: **75.97%**

### Final Model: `RandomForestClassifier`
Hyperparameters:
```python
{
  'n_estimators': 100,
  'max_depth': 10,
  'min_samples_split': 2
}

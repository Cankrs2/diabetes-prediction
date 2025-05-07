import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

df = pd.read_csv('/Users/cengiz/Desktop/Veri Bilimi Kampi/Diabetes-Prediction/diabetes.csv')
print(df.isnull().sum())

columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in columns_with_zeros:
    df[col] = df[col].replace(0,np.nan)
df.fillna(df.mean(), inplace=True)
df['AgeGroup'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60, 80], labels=False)
df['BMI_Glucose_Ratio'] = df['BMI'] / (df['Glucose'] + 1)  # +1 bölme hatası olmasın diye


df.fillna(df.mean(), inplace=True)
X = df.drop(columns= ['Outcome'])
y = df['Outcome']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Logistic Regression
model = LogisticRegression()
model.fit(X_train_scaled,y_train)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_pred,y_test)
print(accuracy)

#Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled,y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_pred_rf,y_test)
print(accuracy_rf)



param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_scaled, y_train)

print("En iyi parametreler:", grid.best_params_)
print("En iyi doğruluk (cv):", grid.best_score_)

# Test verisi ile test edelim
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
print("Test doğruluğu:", accuracy_score(y_test, y_pred_best))



# Task 1 - Supervised Learning: Forest Fires Dataset
# BS Artificial Intelligence _  Machine Learning Assignment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv('forestfires.csv')
print("Dataset shape:", df.shape)

# Encode categorical columns
le = LabelEncoder()
df['month'] = le.fit_transform(df['month'])
df['day'] = le.fit_transform(df['day'])

# Features and target
X = df.drop('area', axis=1)
y = np.log1p(df['area'])  # log(1 + area) to handle skewed distribution

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and back-transform
y_pred_log = model.predict(X_test)
y_pred_actual = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)

# Evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae = mean_absolute_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test, y_pred_log)

print("\n=== Task 1: Supervised Learning Results ===")
print(f"RMSE: {rmse:.2f} hectares")
print(f"MAE:  {mae:.2f} hectares")
print(f"R2:   {r2:.4f}")

# Feature importance plot
importances = model.feature_importances_
features = df.drop('area', axis=1).columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.bar(range(len(features)), importances[indices],
        color='steelblue', edgecolor='white')
plt.xticks(range(len(features)),
           [features[i] for i in indices], rotation=45, ha='right')
plt.title('Feature Importances - Random Forest Regressor (Forest Fires Dataset)')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved as feature_importance.png")

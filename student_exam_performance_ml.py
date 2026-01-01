# Student Exam Performance Prediction Using Machine Learning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Generate dataset
np.random.seed(42)

study_hours = np.random.randint(1, 8, 300)
attendance = np.random.randint(60, 101, 300)
sleep_hours = np.random.randint(4, 9, 300)
internal_marks = np.random.randint(30, 91, 300)
internet_usage = np.random.randint(1, 6, 300)
parental_education = np.random.randint(1, 6, 300)

final_marks = (
    0.4 * study_hours * 5
    + 0.2 * attendance * 0.5
    + 0.15 * sleep_hours * 4
    + 0.15 * internal_marks
    - 0.1 * internet_usage * 3
    + 0.1 * parental_education * 4
    + np.random.normal(0, 5, 300)
)

final_marks = np.clip(final_marks, 0, 100)

data = pd.DataFrame({
    "Study Hours": study_hours,
    "Attendance": attendance,
    "Sleep Hours": sleep_hours,
    "Internal Marks": internal_marks,
    "Internet Usage": internet_usage,
    "Parental Education": parental_education,
    "Final Marks": final_marks
})

print(data.head())

# Split data
X = data.drop("Final Marks", axis=1)
y = data["Final Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Results
print("\nLinear Regression Results")
print("MSE:", mean_squared_error(y_test, lr_pred))
print("R2 Score:", r2_score(y_test, lr_pred))

print("\nRandom Forest Results")
print("MSE:", mean_squared_error(y_test, rf_pred))
print("R2 Score:", r2_score(y_test, rf_pred))

# Plot
plt.scatter(y_test, rf_pred)
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Student Marks")
plt.grid()
plt.show()

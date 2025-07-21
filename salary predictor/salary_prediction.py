import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('salary_data.csv')

# Encode 'Gender' column
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])  # Male=1, Female=0

# Define features and target
X = data[['YearsExperience', 'Age', 'Gender']]
y = data['Salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ========== Linear Regression ==========
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
print("\n[Linear Regression]")
print(f"MSE: {mean_squared_error(y_test, lr_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, lr_pred):.2f}")

# ========== Logistic Regression (classification) ==========
# Binary classification: salary above/below median
y_class = (y > y.median()).astype(int)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_class, test_size=0.2, random_state=0)

log_model = LogisticRegression()
log_model.fit(X_train_cls, y_train_cls)
log_pred = log_model.predict(X_test_cls)
log_acc = (log_pred == y_test_cls).mean()
print("\n[Logistic Regression - Salary > Median Classification]")
print(f"Accuracy: {log_acc:.2f}")

# ========== Random Forest Regressor ==========
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("\n[Random Forest Regressor]")
print(f"MSE: {mean_squared_error(y_test, rf_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, rf_pred):.2f}")

# ========== User Prediction ==========
print("\n--- Predict Salary Using Random Forest ---")
exp = float(input("Enter years of experience: "))
age = int(input("Enter your age: "))
gender_input = input("Enter gender (Male/Female): ").strip().lower()
gender = 1 if gender_input == 'male' else 0

user_input = [[exp, age, gender]]
predicted_salary = rf_model.predict(user_input)
print(f"Predicted Salary (Random Forest): ₹{predicted_salary[0]:.2f}")

# ========== Plotting Comparison ==========
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual', marker='o')
plt.plot(lr_pred, label='Linear Regression', marker='x')
plt.plot(rf_pred, label='Random Forest', marker='s')
plt.title("Actual vs Predicted Salary")
plt.xlabel("Test Sample Index")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

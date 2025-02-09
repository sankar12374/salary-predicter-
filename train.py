import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
df = pd.read_csv("random_dataset.csv")

# Check if dataset is loaded properly
if df.empty:
    print("❌ Error: Dataset is empty!")
    exit()

# Encode categorical column 'Job Role'
label_encoder = LabelEncoder()
df["Job Role"] = label_encoder.fit_transform(df["Job Role"])

# Define Features (X) and Target (y)
X = df[["Age", "Job Role"]]  # Inputs (Age and Job Role)
y = df["Salary (INR)"]       # Output (Salary)

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)  # ✅ Now X_train and y_train are properly defined

print("✅ Model Training Complete!")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Save model for future use
import joblib
joblib.dump(model, "salary_prediction_model.pkl")
print("✅ Model saved as 'salary_prediction_model.pkl'")

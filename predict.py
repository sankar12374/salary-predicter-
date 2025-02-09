import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("salary_prediction_model.pkl")
print("✅ Model Loaded Successfully!")

# Load dataset to get Job Role encoding
df = pd.read_csv("large_dataset.csv")

# Encode Job Roles to match training data
label_encoder = LabelEncoder()
df["Job Role"] = label_encoder.fit_transform(df["Job Role"])

# Get all job roles (converted to lowercase for case-insensitive comparison)
job_roles_dict = {job.lower(): job for job in label_encoder.classes_}

# Get user input
age = int(input("Enter Age: "))
job_role_input = input("Enter Job Role: ").strip().lower()  # Convert input to lowercase

# Convert Job Role to encoded value
if job_role_input in job_roles_dict:
    job_encoded = label_encoder.transform([job_roles_dict[job_role_input]])[0]
else:
    print(f"❌ Error: '{job_role_input}' is not a recognized job role.")
    print("✅ Available job roles:", list(job_roles_dict.keys()))  # Show available roles
    exit()

# Prepare input data
new_data = np.array([[age, job_encoded]])

# Make Prediction
predicted_salary = model.predict(new_data)
print(f"💰 Predicted Salary for Age {age}, Job Role '{job_roles_dict[job_role_input]}': {predicted_salary[0]:,.2f} INR")

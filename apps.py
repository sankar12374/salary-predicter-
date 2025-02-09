from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("salary_prediction_model.pkl")

# Load dataset to get Job Role encoding
df = pd.read_csv("random_dataset.csv")

# Encode job roles
label_encoder = LabelEncoder()
df["Job Role"] = label_encoder.fit_transform(df["Job Role"])
print("Recognized Job Roles:", label_encoder.classes_)

# Store job roles dictionary for validation
job_roles_dict = {job.lower(): job for job in label_encoder.classes_}

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = int(request.form["age"])
        job_role_input = request.form["job_role"].strip().lower()

        # Convert job role input to correct format
        if job_role_input in job_roles_dict:
            job_encoded = label_encoder.transform([job_roles_dict[job_role_input]])[0]
        else:
            return jsonify({"error": f"Invalid job role '{job_role_input}'. Try again!"})

        # Prepare input data
        new_data = np.array([[age, job_encoded]])

        # Make Prediction
        predicted_salary = model.predict(new_data)[0]

        return jsonify({"salary": f"{predicted_salary:,.2f} INR"})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

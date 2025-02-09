import pandas as pd
import random

# Sample Data Options
names = ["Aarav", "Vihaan", "Saanvi", "Ishaan", "Ananya", "Rohan", "Kiara", "Dev", "Meera", "Kabir"]
cities = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Surat"]
job_roles = ["Software Engineer", "Data Analyst", "Doctor", "Teacher", "Accountant", "Marketing Manager",
             "HR", "Sales Executive", "Graphic Designer", "Research Scientist"]

# Number of records to generate
num_records = 1000  # Change this to any large number

# Generate Random Data
data = {
    "ID": list(range(1, num_records + 1)),
    "Name": [random.choice(names) for _ in range(num_records)],
    "Age": [random.randint(18, 60) for _ in range(num_records)],
    "City": [random.choice(cities) for _ in range(num_records)],
    "Salary (INR)": [random.randint(20000, 150000) for _ in range(num_records)],  # Expanded salary range
    "Job Role": [random.choice(job_roles) for _ in range(num_records)]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("large_dataset.csv", index=False)
print(f"✅ Large dataset with {num_records} rows created successfully as 'large_dataset.csv'!")

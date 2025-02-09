import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("large_dataset.csv")  # Ensure this file exists in your working directory

# Check if DataFrame loaded correctly
print("✅ Dataset Loaded Successfully!")
print(df.head())  # Show first few rows

# Check if "Job Role" column exists
if "Job Role" not in df.columns:
    print("❌ Error: 'Job Role' column is missing!")
    exit()

# Check for missing values and fill them if needed
df["Job Role"].fillna("Unknown", inplace=True)

# Convert 'Job Role' to numerical labels
label_encoder = LabelEncoder()
df["Job Role"] = label_encoder.fit_transform(df["Job Role"])

print("✅ Job Role successfully encoded!")
print(df.head())  # Show transformed dataset

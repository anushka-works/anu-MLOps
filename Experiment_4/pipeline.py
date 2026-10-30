import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

# Read dataset
df = pd.read_csv("data/raw/dataset.csv")

# Remove duplicate records
df = df.drop_duplicates()

# Separate input and output
X = df.drop("purchased", axis=1)
y = df["purchased"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Numerical preprocessing
num = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler())

# Categorical preprocessing
cat = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

# Combine preprocessing
pre = ColumnTransformer([
    ("num", num, ["age", "income"]),
    ("cat", cat, ["department"])
])

# Apply preprocessing
X_train = pre.fit_transform(X_train)
X_test = pre.transform(X_test)

# Save pipeline
os.makedirs("data/processed", exist_ok=True)
joblib.dump(pre, "data/processed/preprocessor.pkl")

print("Pipeline executed successfully!")
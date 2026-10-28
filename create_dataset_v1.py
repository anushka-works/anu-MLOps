
import pandas as pd
import os

data = {
    "Name": ["Amit", "Neha", "Rahul", "Priya", "Sneha"],
    "Marks": [85, 90, 78, 92, 88]
}

df = pd.DataFrame(data)

os.makedirs("data", exist_ok=True)
df.to_csv("data/raw_data.csv", index=False)

print("Dataset created successfully!")
print(df)

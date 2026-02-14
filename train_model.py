import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

df = pd.read_csv("dataset.csv")

if "disease" not in df.columns:
    raise ValueError("Column 'disease' not found in dataset.csv")

for col in df.columns:
    if df[col].dtype == "object" and col != "disease":
        df = df.drop(col, axis=1)

X = df.drop("disease", axis=1)
y = df["disease"]

X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

to_save = {
    "model": model,
    "symptoms": list(X.columns)
}

with open("model.pkl", "wb") as f:
    pickle.dump(to_save, f)

print("Model trained and saved successfully.")

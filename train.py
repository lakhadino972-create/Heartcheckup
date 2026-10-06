import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("dataset_heart.csv")

# Rename columns to match the keys the frontend sends
df.columns = ["age", "sex", "chest_pain_type", "resting_blood_pressure",
              "serum_cholestoral", "fasting_blood_sugar", "resting_ecg",
              "max_heart_rate", "exercise_induced_angina", "oldpeak",
              "st_segment", "major_vessels", "thal", "target"]

# Dataset uses 1 = no disease, 2 = disease. Convert to 0/1.
df["target"] = (df["target"] == 2).astype(int)

X = df.drop(columns="target")
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# The scaler lives inside the pipeline, so it is saved with the model
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Save the model together with the column order
joblib.dump({"model": model, "features": list(X.columns)}, "model.pkl")
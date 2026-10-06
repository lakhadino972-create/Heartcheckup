import joblib, pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

bundle = joblib.load("model.pkl")
model, features = bundle["model"], bundle["features"]

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.post("/predict")
def predict(data: dict):
    row = pd.DataFrame([data])[features]      # same column order as training
    proba = float(model.predict_proba(row)[0][1])
    return {"prediction": int(proba >= 0.5), "probability": proba}
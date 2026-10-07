import joblib, pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

bundle = joblib.load("model.pkl")
model, features = bundle["model"], bundle["features"]

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/predict")
def predict(data: dict):
    row = pd.DataFrame([data])[features]
    proba = float(model.predict_proba(row)[0][1])
    return {"prediction": int(proba >= 0.5), "probability": proba}
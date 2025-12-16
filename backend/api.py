from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Cardio Risk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cardio-risk-prediction.streamlit.app"
        "localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "cardio_model_knn.pkl")
knn_model = joblib.load(model_path)


class CardioInput(BaseModel):
    age: int
    gender: int
    height: float
    weight: float
    ap_hi: float
    ap_lo: float
    cholesterol: int
    gluc: int
    smoke: int
    alco: int
    active: int


@app.get("/")
def health():
    return {"status": "API running"}


@app.post("/predict")
def predict_risk(data: CardioInput):
    d = data.dict()
    bmi = d["weight"] / ((d["height"] / 100) ** 2)

    input_df = pd.DataFrame([{**d, "bmi": bmi}])

    proba = float(knn_model.predict_proba(input_df)[0][1])
    pred = int(knn_model.predict(input_df)[0])

    return {
        "model_used": "KNN",
        "prediction": pred,
        "risk_label": "High" if pred == 1 else "Low",
        "probability": proba
    }

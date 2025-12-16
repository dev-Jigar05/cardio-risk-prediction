from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Cardio Risk API")

# lr_model = joblib.load("cardio_model_lr.pkl")
# rf_model = joblib.load("cardio_model_rf.pkl")
knn_model = joblib.load("cardio_model_knn.pkl")

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


@app.post("/predict")
def predict_risk(
    data: CardioInput,
    model_name: str = "lr" 
):
    
    # if model_name == "rf":
    #     used_model = rf_model
    # else:
    #     used_model = lr_model

    used_model = knn_model

    d = dict(data)

    bmi = d["weight"] / ((d["height"] / 100) ** 2)

    input_df = pd.DataFrame([{
        "age": d["age"],
        "gender": d["gender"],
        "height": d["height"],
        "weight": d["weight"],
        "ap_hi": d["ap_hi"],
        "ap_lo": d["ap_lo"],
        "cholesterol": d["cholesterol"],
        "gluc": d["gluc"],
        "smoke": d["smoke"],
        "alco": d["alco"],
        "active": d["active"],
        "bmi": bmi
    }])

    proba = float(used_model.predict_proba(input_df)[0][1])
    pred = int(used_model.predict(input_df)[0])

    return {
        "model_used": "Random Forest" if model_name == "rf" else "Logistic Regression",
        "prediction": pred,
        "risk_label": "High" if pred == 1 else "Low",
        "probability": proba
    }

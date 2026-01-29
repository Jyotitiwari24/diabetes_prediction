import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load model + scaler
model = joblib.load("model/diabetes_model.pkl")
scaler = joblib.load("model/scaler.pkl")

app = FastAPI(title="Diabetes Prediction API")


class PatientData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float


@app.get("/")
def home():
    return {"message": "Diabetes Prediction API Running"}


@app.post("/predict")
def predict(data: PatientData):
    input_data = np.array([[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]])

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    return {
        "prediction": int(prediction),
        "result": result
    }

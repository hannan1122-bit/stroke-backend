from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
import os

# Load model
model_path = os.environ.get("MODEL_PATH", "stroke_pipeline_knn.pkl")
model = joblib.load(model_path)

class PatientData(BaseModel):
    gender: int
    age: float
    hypertension: int
    heart_disease: int
    ever_married: int
    Residence_type: int
    avg_glucose_level: float
    bmi: float
    work_type_Never_worked: int
    work_type_Private: int
    work_type_Self_employed: int
    work_type_children: int
    smoking_status_formerly_smoked: int  
    smoking_status_never_smoked: int
    smoking_status_smokes: int

app = FastAPI()

# Add CORS middleware
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://YOUR-VERCEL-APP.vercel.app",  # production frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict(data: PatientData):
    input_dict = {
        'gender': data.gender,
        'age': data.age,
        'hypertension': data.hypertension,
        'heart_disease': data.heart_disease,
        'ever_married': data.ever_married,
        'Residence_type': data.Residence_type,
        'avg_glucose_level': data.avg_glucose_level,
        'bmi': data.bmi,
        'work_type_Never_worked': data.work_type_Never_worked,
        'work_type_Private': data.work_type_Private,
        'work_type_Self-employed': data.work_type_Self_employed,
        'work_type_children': data.work_type_children,
        'smoking_status_formerly smoked': data.smoking_status_formerly_smoked,
        'smoking_status_never smoked': data.smoking_status_never_smoked,
        'smoking_status_smokes': data.smoking_status_smokes
    }
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    return {"prediction": int(prediction)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

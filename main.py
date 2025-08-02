from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Literal
import joblib
import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

app = FastAPI(title="Netflix Customer Churn Prediction")

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

MODEL_PATH = os.path.join("models", "RandomForest_best_model.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Train the model first.")
model = joblib.load(MODEL_PATH)
logging.info(f"Model loaded from {MODEL_PATH}")


class CustomerData(BaseModel):
    watch_hours: float
    last_login_days: int
    number_of_profiles: int
    avg_watch_time_per_day: float
    subscription_type: Literal["Basic", "Standard", "Premium"]
    payment_method: Literal["Credit Card", "Debit Card", "PayPal", "UPI", "Gift Card"]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})


@app.post("/predict", response_class=HTMLResponse)
async def predict_form(request: Request,
                       watch_hours: float = Form(...),
                       last_login_days: int = Form(...),
                       number_of_profiles: int = Form(...),
                       avg_watch_time_per_day: float = Form(...),
                       subscription_type: str = Form(...),
                       payment_method: str = Form(...)):

    try:
        input_data = pd.DataFrame([{
            "watch_hours": watch_hours,
            "last_login_days": last_login_days,
            "number_of_profiles": number_of_profiles,
            "avg_watch_time_per_day": avg_watch_time_per_day,
            "subscription_type": subscription_type,
            "payment_method": payment_method
        }])

        # Predict
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        result_text = f"Customer is {'likely to Churn' if prediction == 1 else 'not likely to Churn'}"

        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": result_text,
            "probability": round(probability, 4)
        })

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/api/predict")
async def predict_api(data: CustomerData):
    try:
        input_data = pd.DataFrame([data.model_dump()])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        return {
            "prediction": int(prediction),
            "message": "Customer is likely to Churn" if prediction == 1 else "Customer is not likely to Churn",
            "probability": round(probability, 4)
        }
    except Exception as e:
        logging.error(f"API prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

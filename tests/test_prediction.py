import pytest
import joblib
import os
import pandas as pd

def test_model_prediction():
    model_path = "models/RandomForest_best_model.pkl"
    assert os.path.exists(model_path), "Model file should exist"

    model = joblib.load(model_path)
    input_data = pd.DataFrame([{
        "watch_hours": 10,
        "last_login_days": 3,
        "number_of_profiles": 2,
        "avg_watch_time_per_day": 2.5,
        "subscription_type": "basic",
        "payment_method": "credit"
    }])

    pred = model.predict(input_data)
    assert pred[0] in [0, 1], "Prediction should be 1 or 0"

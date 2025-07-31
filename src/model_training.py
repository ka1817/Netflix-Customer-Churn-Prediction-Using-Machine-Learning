import os
import joblib
import logging
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessor
logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def train_and_log_models():

    df = DataIngestion().load_data()

    numerical_features = ['watch_hours', 'last_login_days', 'number_of_profiles', 'avg_watch_time_per_day']
    categorical_features = ['subscription_type', 'payment_method']
    target_column = 'churned'

    preprocessor = DataPreprocessor(numerical_features, categorical_features, target_column)
    preprocessor.build_pipeline()

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "SVC": SVC(probability=True),
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
    }

    param_grids = {
        "RandomForest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10],
            "classifier__min_samples_split": [2, 5],
        },
        "SVC": {
            "classifier__C": [0.1, 1],
            "classifier__kernel": ["rbf", "linear"],
            "classifier__gamma": ["scale", "auto"]
        },
        "GradientBoosting": {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.01, 0.1],
            "classifier__max_depth": [3, 5]
        }
    }

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Netflix Churn Models")

    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)

    for name, model in models.items():
        logging.info(f"Training {name}...")

        pipeline = Pipeline([
            ('preprocessor', preprocessor.pipeline),
            ('classifier', model)
        ])

        grid = GridSearchCV(pipeline, param_grid=param_grids[name], cv=5, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0

        logging.info(f"{name} | Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC AUC: {auc:.4f}")

        with mlflow.start_run(run_name=name):
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics({
                "accuracy": acc,
                "f1_score": f1,
                "roc_auc": auc
            })
            mlflow.sklearn.log_model(best_model, name)

        model_path = os.path.join(models_dir, f"{name}_best_model.pkl")
        joblib.dump(best_model, model_path)
        logging.info(f"{name} best model saved at: {model_path}")
if __name__ == "__main__":
    train_and_log_models()

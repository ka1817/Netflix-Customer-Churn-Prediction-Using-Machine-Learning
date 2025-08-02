import os
import joblib
import logging
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    recall_score,
    classification_report,
    roc_curve
)

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
    mlflow.set_experiment("Netflix Churn Prediction")

    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

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
        recall = recall_score(y_test, y_pred)
        auc_test = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0

        cv_auc_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc')
        auc_cv_mean = np.mean(cv_auc_scores)

        logging.info(f"{name} | Accuracy: {acc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f} | Test ROC AUC: {auc_test:.4f} | CV ROC AUC: {auc_cv_mean:.4f}")

        if auc_test - auc_cv_mean > 0.02:  
            logging.warning(f"⚠ {name} may be overfitting! Test AUC is significantly higher than CV AUC.")

        class_report = classification_report(y_test, y_pred)
        report_path = os.path.join(reports_dir, f"{name}_classification_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Model: {name}\n\n")
            f.write("Best Parameters:\n")
            f.write(str(grid.best_params_))
            f.write("\n\nClassification Report:\n")
            f.write(class_report)
            f.write(f"\n\nCross-Validation AUC: {auc_cv_mean:.4f}\n")
            f.write(f"Test AUC: {auc_test:.4f}\n")
        logging.info(f"Classification report saved at: {report_path}")

        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure()
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_test:.2f})')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}')
            plt.legend(loc='lower right')

            roc_path = os.path.join(reports_dir, f"{name}_roc_curve.png")
            plt.savefig(roc_path)
            plt.close()
            logging.info(f"ROC curve saved at: {roc_path}")
        else:
            roc_path = None

        with mlflow.start_run(run_name=name):
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics({
                "accuracy": acc,
                "f1_score": f1,
                "recall": recall,
                "roc_auc_test": auc_test,
                "roc_auc_cv": auc_cv_mean
            })
            mlflow.sklearn.log_model(best_model, name)
            mlflow.log_artifact(report_path)
            if roc_path:
                mlflow.log_artifact(roc_path)

        model_path = os.path.join(models_dir, f"{name}_best_model.pkl")
        joblib.dump(best_model, model_path)
        logging.info(f"{name} best model saved at: {model_path}")

if __name__ == "__main__":
    train_and_log_models()

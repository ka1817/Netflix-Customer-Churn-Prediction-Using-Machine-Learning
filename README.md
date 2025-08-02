# 📊 Netflix Customer Churn Prediction

## 🚀 Project Overview

Netflix, like many subscription-based platforms, faces the challenge of customer churn. Retaining existing customers is significantly more cost-effective than acquiring new ones. This project delivers a full-scale machine learning solution to predict customer churn using behavioral and subscription data, from ingestion to deployment via a FastAPI interface.

This repository presents a production-grade, explainable, and reproducible ML pipeline with CI/CD, experiment tracking (**MLflow**), data versioning (**DVC**), and containerized deployment using **Docker**.

---

## 🎯 Problem Statement

Netflix seeks to proactively identify users likely to cancel their subscriptions. Predicting churn enables targeted interventions to retain users and minimize revenue loss.

> **Goal:** Build an ML classification model that predicts churn based on customer behavior and plan details.

---

## 📌 Key Features Used

| Feature                    | Type        | Description                                    |
| -------------------------- | ----------- | ---------------------------------------------- |
| watch\_hours               | Numerical   | Total hours watched                            |
| last\_login\_days          | Numerical   | Days since last login                          |
| number\_of\_profiles       | Numerical   | Total profiles under the account               |
| avg\_watch\_time\_per\_day | Numerical   | Daily average watch time                       |
| subscription\_type         | Categorical | Subscription level: Basic, Standard, Premium   |
| payment\_method            | Categorical | Payment method: Credit Card, UPI, PayPal, etc. |
| churned                    | Target      | 1 = Churned, 0 = Not churned                   |

---

## 📊 Key EDA Insights

### 🔬 Feature Significance

| Feature                    | Test           | p-value | Significant? |
| -------------------------- | -------------- | ------- | ------------ |
| subscription\_type         | Chi-Square     | 0.0000  | ✅ Yes        |
| payment\_method            | Chi-Square     | 0.0000  | ✅ Yes        |
| number\_of\_profiles       | Chi-Square     | 0.0000  | ✅ Yes        |
| watch\_hours               | Mann-Whitney U | 0.0000  | ✅ Yes        |
| last\_login\_days          | Mann-Whitney U | 0.0000  | ✅ Yes        |
| avg\_watch\_time\_per\_day | Mann-Whitney U | 0.0000  | ✅ Yes        |
| age                        | Mann-Whitney U | 0.7803  | ❌ No         |
| gender, region, device     | Chi-Square     | > 0.3   | ❌ No         |

> ✅ These statistically significant features were included in the final model pipeline.

---

## 🏗️ Project Architecture

```bash
netflix-churn-prediction/
├── data/                     # Raw and processed data
├── models/                   # Trained model binaries
├── reports/                  # Classification reports & plots
├── static/                   # CSS
├── templates/                # HTML UI
├── src/
│   ├── data_ingestion.py     # Load dataset
│   ├── data_preprocessing.py # Pipeline for scaling & encoding
│   └── model_training.py     # ML training & evaluation
├── main.py                   # FastAPI backend
├── Dockerfile                # Containerization
├── .dvc/                     # DVC for data version control
├── .github/workflows/        # CI/CD GitHub Actions
└── README.md
```

---

## ⚙️ End-to-End ML Workflow

### 1️⃣ Data Ingestion

* Loads `.csv` into DataFrame
* Handles errors and logs shape/summary

### 2️⃣ Preprocessing

* OneHotEncoding (categorical)
* StandardScaler (numerical)
* Uses `ColumnTransformer` for pipeline modularity

### 3️⃣ Model Training

* Models: `RandomForest`, `GradientBoosting`, `SVC`
* `GridSearchCV` for hyperparameter tuning
* Model artifacts saved to `models/`
* ROC curves + classification reports saved to `reports/`

### 4️⃣ MLflow Tracking ✅

* Tracks experiment metadata, metrics, parameters
* Stores models and artifacts
* UI accessible at `localhost:5000`

---

## 🧪 Model Performance

| Model             | Accuracy | F1 Score | ROC AUC (Test) | ROC AUC (CV) | Notes                         |
| ----------------- | -------- | -------- | -------------- | ------------ | ----------------------------- |
| Random Forest     | 0.99     | 0.99     | **0.9995**     | 0.9987       | ✅ Best overall【13†source】     |
| Gradient Boosting | 0.99     | 0.99     | 0.9989         | 0.9991       | Robust & efficient【12†source】 |
| SVC               | 0.93     | 0.93     | 0.9844         | 0.9822       | Lightweight【14†source】        |

---

## 🌐 FastAPI Deployment

### 🔧 API Endpoints:

* `/`: HTML frontend form for manual input
* `/api/predict`: JSON-based API for programmatic inference

### 🔌 Model Used:

* Random Forest (best AUC + accuracy)
* Accepts form or JSON input
* Returns churn prediction + confidence

---

## 🐳 Docker Setup

```Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Run locally:

```bash
docker build -t netflix-churn .
docker run -p 8000:8000 netflix-churn
```

---

## 🔁 CI/CD Pipeline (GitHub Actions)

### ✅ Stages:

1. **Test Phase**

   * Install dependencies
   * Run `pytest` on unit tests
   * Pull versioned data using `dvc pull`

2. **Build Phase**

   * Docker image build with `CACHEBUST` arg
   * Push to DockerHub using GitHub Secrets

3. **Deploy Phase**

   * SSH into EC2 instance
   * Stop, remove old container
   * Pull and launch updated Docker image

### 🔐 GitHub Repository Secrets

| Name                    | Purpose                            |
| ----------------------- | ---------------------------------- |
| `AWS_ACCESS_KEY_ID`     | AWS auth for DVC S3                |
| `AWS_SECRET_ACCESS_KEY` | AWS auth for DVC S3                |
| `DOCKER_USERNAME`       | DockerHub username for push        |
| `DOCKER_PASSWORD`       | DockerHub password/token           |
| `EC2_HOST`              | Public IP/DNS of EC2 instance      |
| `EC2_USER`              | SSH user for EC2 login             |
| `EC2_SSH_KEY`           | Private SSH key for GitHub Actions |

---

## 🧬 Data Versioning with DVC

* Tracks raw and preprocessed data versions
* Uses `.dvc/config` to connect to **AWS S3** remote
* Run `dvc push` and `dvc pull` to sync across environments
* Ensures reproducibility in CI and local experiments

---

## 📌 Business Value & Insights

* 🧠 **High-risk churn users** are linked to:

  * Low engagement (low watch hours)
  * Infrequent logins
  * Basic plans & non-card payments

* 📈 **Operational Benefits**:

  * Preemptive retention campaigns
  * Personalized offers to vulnerable users
  * Reduce marketing costs via targeted outreach

---

## ✅ Run Locally (No Docker)

```bash
git clone <repo_url>
cd netflix-churn-prediction
python src/model_training.py        # Train all models
uvicorn main:app --reload           # Launch API server
```

---

## 🙌 Author

* 👨‍💻 Katta Sai Pranav Reddy

---

## 📎 Tech Stack

* **Python 3.10**
* **Scikit-learn**, **MLflow**, **DVC**, **FastAPI**, **Docker**
* **GitHub Actions**, **AWS EC2**, **S3 Remote Storage**

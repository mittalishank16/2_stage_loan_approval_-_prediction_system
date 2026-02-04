---
title: Loan_approval_prediction
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---


# Loan Approval & Amount Prediction (MLOps Project)

This project simulates how financial institutions evaluate loan applications using a **two-stage machine learning pipeline**. It is built as an **end-to-end MLOps project** and deployed on **Hugging Face Spaces**.

---

## Project Overview

Financial institutions typically follow a multi-step process when assessing loan applications. This project replicates that workflow using machine learning:

1. **Loan Approval Model (Classification)**  
   Determines whether a loan application should be **approved or rejected**.

2. **Loan Amount Model (Regression)**  
   For approved applicants, predicts the **optimal loan amount** based on applicant and financial features.

Both stages are integrated into a single inference pipeline and exposed through a web interface.

---

## Modeling Approach

### Stage 1: Loan Approval (Classification)
- Predicts whether a loan should be approved
- Outputs: `Approved` / `Not Approved`

### Stage 2: Loan Amount Prediction (Regression)
- Runs **only if the loan is approved**
- Predicts the optimal loan amount
- Prevents unrealistic predictions for rejected applications

This two-stage design mirrors real-world lending systems and improves interpretability.

---

## Tech Stack

- **Python**
- **Pandas, NumPy** – Data processing
- **Scikit-learn** – Model training
- **FastAPI** – Model serving
- **Gradio** – Interactive UI
- **Docker** – Containerization
- **GitHub Actions** – CI/CD
- **Hugging Face Spaces** – Deployment

---

## Project Structure

.
├── app/
│ ├── main.py # FastAPI app
│ ├── inference.py # Inference pipeline
│ └── schemas.py # Request/response schemas
├── gradio_app.py # Gradio UI
├── models/
│ ├── approval_model.pkl
│ └── amount_model.pkl
├── artifacts/
│ ├── encoders.pkl
│ └── feature_schema.json
├── Dockerfile
├── requirements.txt
├── README.md
└── .github/workflows/
└── deploy.yml # CI/CD pipeline

#### run docker-compose down
#### docker-compose up --build

#### docker build -t 2_STAGE_LOAN_SYSTEM .
#### docker run -p 7860:7860 2_stage_loan_system

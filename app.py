from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from src.inference_pipeline.inference import LoanInferencePipeline

app = FastAPI(title="2-Stage Loan Approval API")

# Initialize the pipeline once when the app starts
pipeline = LoanInferencePipeline()

# Schema for incoming single requests
class LoanApplication(BaseModel):
    Gender: str
    Age: float
    Income_USD: float
    Location: str
    Property_Price: float
    Credit_Score: float
    Has_Active_Credit_Card: str
    Property_Location: str
    Co_Applicant: float = 0
    Current_Loan_Expenses_USD: float = 0
    Expense_Type_1: str = "N"
    Expense_Type_2: str = "N"
    Loan_Amount_Request_USD: float = 0
    Profession: str = "Working"

@app.get("/")
def health_check():
    return {"status": "online", "model": "loaded"}

@app.post("/predict")
def predict_loan(data: LoanApplication):
    try:
        # Map Pydantic model to DataFrame matching the expected raw input names
        gender_map = {"Male": "M", "Female": "F"}
        input_dict = {
            'Gender': [gender_map.get(data.Gender, data.Gender)], # Maps "Male" to "M"
            'Age': [data.Age],
            'Income (USD)': [data.Income_USD],
            'Location': [data.Location],
            'Property Price': [data.Property_Price],
            'Credit Score': [data.Credit_Score],
            'Has Active Credit Card': [data.Has_Active_Credit_Card],
            'Property Location': [data.Property_Location],
            'Co-Applicant': [data.Co_Applicant],
            'Current Loan Expenses (USD)': [data.Current_Loan_Expenses_USD],
            'Expense Type 1': [data.Expense_Type_1],
            'Expense Type 2': [data.Expense_Type_2],
            'Loan Amount Request (USD)': [data.Loan_Amount_Request_USD],
            'Profession': [data.Profession]
        }
        
        df = pd.DataFrame(input_dict)
        results = pipeline.predict(df) # Uses stage 1 (clf) and stage 2 (reg)
        return results[0]
    
    except Exception as e:
        # Add these two lines to see the real error in 'docker logs'
        import traceback
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=str(e))
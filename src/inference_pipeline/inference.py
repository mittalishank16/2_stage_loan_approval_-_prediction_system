import sys
import os
import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
from tensorflow.keras.models import load_model

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# Robust Path Resolution
if os.environ.get('DOCKER_ENV') == 'true':
    BASE_DIR = Path("/app")
else:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

sys.path.append(str(BASE_DIR))

from src.ingestion_pipeline.ingestion import (
    impute_999, 
    ltv_transform, 
    filter_professions, 
    apply_binary_encoding, 
    handle_ohe_encoding, 
    apply_transforms
)

# --- Configuration & Path Constants ---
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" 
TEST_DATA_PATH = DATA_DIR / "test.csv"

CLASSIFICATION_MODEL_PATH = MODEL_DIR / "best_randomforest_fine_tuned_classification_model.pkl"
REGRESSION_MODEL_PATH = MODEL_DIR / "best_xgboost_fine_tuned_regression_model.pkl"
# Path to the exported median values
IMPUTATION_VALUES_PATH = MODEL_DIR / "imputation_values.joblib"

class LoanInferencePipeline:
    def __init__(self):
        print("--- Loading Pre-trained Models & Pipeline Artifacts ---")
        self.clf_model = self._load_model(CLASSIFICATION_MODEL_PATH)
        self.reg_model = self._load_model(REGRESSION_MODEL_PATH)
        self.model_features = self.clf_model.feature_names_in_
        
        # --- LOAD EXPORTED MEDIAN VALUES ---
        if IMPUTATION_VALUES_PATH.exists():
            print(f"[+] Loading training medians from: {IMPUTATION_VALUES_PATH.name}")
            self.impute_values = joblib.load(IMPUTATION_VALUES_PATH)
        else:
            print("⚠️ Warning: imputation_values.joblib not found. Using default placeholders.")
            self.impute_values = {
                'Income (USD)': 3000.0,
                'Credit Score': 700.0,
                'Age': 35.0,
                'Property Price': 50000.0
            }

    def _load_model(self, path):
        if not path.exists():
            raise FileNotFoundError(f"Missing model file: {path.absolute()}")
        if path.suffix in ['.h5', '.keras']:
            print(f"[+] Loading Keras: {path.name}")
            return load_model(path)
        elif path.suffix == '.pkl':
            print(f"[+] Loading Sklearn: {path.name}")
            return joblib.load(path)
        return None

    def preprocess_input(self, raw_data: pd.DataFrame):
        df = raw_data.copy()
        
        # 1. Initial Cleaning
        df.drop(columns=['Customer ID', 'Name', 'Property ID', 'Type of Employment', 'Income Stability'], 
                errors='ignore', inplace=True)
        df = impute_999(df) 

        # 2. Binary Missing Indicators
        for col in ['Income (USD)', 'Credit Score', 'Property Price']:
            if col in df.columns:
                df[f'{col}_was_missing'] = df[col].isna().astype(int)

        # 3. Impute Numerical Columns (Using Loaded Medians)
        for col, value in self.impute_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)

        # 4. Impute Categorical Columns (Mode Imputation)
        categorical_cols = ['Gender', 'Income Stability', 'Location', 'Expense Type 1', 
                            'Expense Type 2', 'Has Active Credit Card', 'Property Location']
        
        for col in categorical_cols:
            if col in df.columns:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else ""
                df[col] = df[col].fillna(mode_val)

        # 5. Feature Engineering
        df = filter_professions(df)
        df = ltv_transform(df)
        
        # 6. Encoding & Scaling
        df = apply_binary_encoding(df)
        df = handle_ohe_encoding(df)
        df = apply_transforms(df)
        
        # 7. Alignment
        for col in self.model_features:
            if col not in df.columns:
                df[col] = 0
        
        df = df[self.model_features]

        # Final safety check
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        return df

    def predict(self, input_data: pd.DataFrame):
        processed_X = self.preprocess_input(input_data)
        
        # Stage 1: Classification
        if hasattr(self.clf_model, 'predict_classes') or 'Sequential' in str(type(self.clf_model)):
            is_approved = (self.clf_model.predict(processed_X) > 0.5).astype(int).flatten()
        else:
            is_approved = self.clf_model.predict(processed_X)
            
        results = []
        for i, status in enumerate(is_approved):
            decision = {
                "status": "APPROVED" if status == 1 else "NOT APPROVED",
                "loan_approved": bool(status)
            }
            
            # Stage 2: Regression (Conditional)
            if status == 1:
                amount_pred = self.reg_model.predict(processed_X.iloc[[i]])
                val = amount_pred[0][0] if len(getattr(amount_pred, 'shape', ())) > 1 else amount_pred[0]
                decision["predicted_sanction_amount"] = f"${float(np.round(val, 2)):,.2f}"
            else:
                decision["predicted_sanction_amount"] = "N/A"
            
            results.append(decision)
            
        return results

if __name__ == "__main__":
    if not TEST_DATA_PATH.exists():
        print(f"Error: Could not find data at {TEST_DATA_PATH}")
    else:
        test_df = pd.read_csv(TEST_DATA_PATH)
        pipeline = LoanInferencePipeline()
        inference_results = pipeline.predict(test_df)
        
        print("\n" + "="*60)
        print(f"{'ROW':<5} | {'DECISION':<15} | {'SANCTIONED AMOUNT':<20}")
        print("-" * 60)
        
        for i, res in enumerate(inference_results):
            if res["loan_approved"]:
                print(f"{i:<5} | {res['status']:<15} | {res['predicted_sanction_amount']:<20}")
            else:
                print(f"{i:<5} | {res['status']:<15} | {'---':<20}")
        
        print("="*60)
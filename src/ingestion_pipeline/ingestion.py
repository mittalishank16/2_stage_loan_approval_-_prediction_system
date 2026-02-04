import os
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, PowerTransformer

# --- Configuration ---
warnings.filterwarnings('ignore')
DATA_DIR = Path("data")
RAW_DATA_PATH = DATA_DIR / "loan_processed_data.csv"
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Ensure directory exists
PROCESSED_DATA_DIR = DATA_DIR / "feature_engineered_pipeline"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. Cleaning & Imputation Functions
# ==========================================

def export_imputation_values(df):
    """
    Calculates medians from the training data and saves them to an artifact.
    These are used during inference to ensure consistency.
    """
    features = ['Income (USD)', 'Credit Score', 'Age', 'Property Price']
    impute_dict = {}
    
    # We use a temporary df to calculate clean medians (handling -999 first)
    temp_df = df.copy()
    for col in features:
        if col in temp_df.columns:
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').replace([-999, 0], np.nan)
            impute_dict[col] = float(temp_df[col].median())
    
    artifact_path = OUTPUT_DIR / 'imputation_values.joblib'
    joblib.dump(impute_dict, artifact_path)
    print(f"✅ Exported training medians to {artifact_path}")
    return impute_dict

def impute_999(df):
    """Replaces -999 placeholders and handles Property Price specific median."""
    df = df.copy()
    features_with_999 = ['Co-Applicant', 'Current Loan Expenses (USD)', 
                         'Loan Sanction Amount (USD)', 'Property Price']

    for col in features_with_999:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').replace([-999, 0], np.nan)
    
    # Specific logic for Property Price
    df['Property Price'] = df['Property Price'].replace(0, np.nan)
    median_price = df['Property Price'].median()
    df['Property Price'] = df['Property Price'].fillna(median_price)
    return df

def filter_professions(df):
    df = df.copy()
    # Define the core professions your model should care about
    # All others (like 'Maternity leave') will be grouped into 'Other'
    top_professions = ['Working', 'Commercial associate', 'Pensioner', 'State servant']
    
    if 'Profession' in df.columns:
        df['Profession'] = df['Profession'].apply(
            lambda x: x if x in top_professions else 'Other'
        )
    return df

def impute_missing_values(df):
    """Standard imputation for general features."""
    df = df.copy()
    mode_features = ['Gender', 'Dependents', 
                     'Has Active Credit Card', 'Property Location']
    median_features = ['Age', 'Income (USD)', 'Current Loan Expenses (USD)', 'Credit Score']
    
    for ft in mode_features:
        if ft in df.columns and not df[ft].mode().empty:
            df[ft] = df[ft].fillna(df[ft].mode()[0])
            
    for ft in median_features:
        if ft in df.columns:
            df[ft] = df[ft].fillna(df[ft].median())
    return df

def remove_outliers_iqr(df):
    """Filters extreme noise to improve model stability."""
    outlier_ft = ['Income (USD)', 'Current Loan Expenses (USD)', 'Property Price']
    existing_ft = [ft for ft in outlier_ft if ft in df.columns]
    
    Q1 = df[existing_ft].quantile(0.25)
    Q3 = df[existing_ft].quantile(0.75)
    IQR = Q3 - Q1
    
    is_outlier = ((df[existing_ft] < (Q1 - 1.5 * IQR)) | 
                  (df[existing_ft] > (Q3 + 1.5 * IQR))).any(axis=1)
    return df[~is_outlier]

# ==========================================
# 2. Transformation & Encoding Functions
# ==========================================

def binary_encoding(df):
    df = df.copy()
    binary_cols = ['Gender', 'Expense Type 1', 'Expense Type 2', 'loan_approval']
    artifact_path = OUTPUT_DIR / 'label_encoders.joblib'
    
    encoders = {}
    for col in binary_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
    joblib.dump(encoders, artifact_path)
    return df

def apply_binary_encoding(df_test):
    """
    Applies the pre-trained LabelEncoders to a test/new dataframe.
    """
    # 1. Path to your saved artifact
    artifact_path = OUTPUT_DIR / 'label_encoders.joblib'
    if not artifact_path.exists():
        raise FileNotFoundError("Run training first to generate encoders.")
    encoders = joblib.load(artifact_path)


    # 3. Apply each encoder to its corresponding column
    df_test = df_test.copy()
    # We use .transform() to apply the existing mapping
    # .astype(str) ensures we don't crash on NaNs or mixed types
    for col, le in encoders.items():
        if col in df_test.columns:
            valid_classes = set(le.classes_)
            df_test[col] = df_test[col].astype(str).apply(
                lambda x: x if x in valid_classes else le.classes_[0]
            )
            df_test[col] = le.transform(df_test[col].astype(str))
    return df_test

def handle_ohe_encoding(df):
    """
    Handles One-Hot Encoding for multi-class columns.
    Saves the final feature list to ensure column alignment during inference.
    """
    df = df.copy()
    ohe_cols = ['Profession', 'Location', 'Has Active Credit Card', 'Property Location']
    df = pd.get_dummies(df, columns=[c for c in ohe_cols if c in df.columns], drop_first=True, dtype=int)
    return df

def apply_scaling_and_power(df):
    scale_cols = ['Age', 'Income (USD)', 'Current Loan Expenses (USD)', 'Credit Score']
    scaler = MinMaxScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    joblib.dump(scaler, OUTPUT_DIR / 'scaler.pkl')
    
    """Applies PowerTransformer and MinMaxScaler, saving objects for inference."""
    # Power Transform
    skew_cols = ['Income (USD)', 'Current Loan Expenses (USD)']
    pt = PowerTransformer(method='yeo-johnson')
    df[skew_cols] = pt.fit_transform(df[skew_cols])
    joblib.dump(pt, OUTPUT_DIR / 'power_transformer.pkl')
    return df

def apply_transforms(vdf):

    # Define mapping of artifact name to its corresponding columns
    vdf = vdf.copy()
    config = {
        'scaler.pkl': ['Age', 'Income (USD)', 'Current Loan Expenses (USD)', 'Credit Score'],
        'power_transformer.pkl': ['Income (USD)', 'Current Loan Expenses (USD)']
    }
    for file, cols in config.items():
        path = OUTPUT_DIR / file
        if path.exists():
            transformer = joblib.load(path)
            vdf[cols] = transformer.transform(vdf[cols])
    return vdf

def ltv_transform(df):
    """Final feature engineering before export."""
    df = df.copy()
    if 'Loan Amount Request (USD)' in df.columns and 'Property Price' in df.columns:
        df['LTV_Ratio'] = df['Loan Amount Request (USD)'] / df['Property Price']
        df['LTV_Ratio'] = df['LTV_Ratio'].replace([np.inf, -np.inf], 0).fillna(0)
    return df.drop(columns=['Loan Amount Request (USD)', 'Property Price'], errors='ignore')

# ==========================================
# 3. The Final Pipeline Runner
# ==========================================

def run_ingestion_pipeline(raw_path):
    """
    The orchestrator that runs the entire sequence from raw data 
    to processed CSVs ready for training.
    """
    print("🚀 Starting Data Ingestion Pipeline...")
    df = pd.read_csv(raw_path)
    df = df.drop(columns=['Customer ID', 'Name', 'Property ID', 'Type of Employment', 'Profession_Pensioner', 'Property Age', 'Income Stability'], errors='ignore')
    
    # Initial Clean for statistics
    df = impute_999(df)
    
    # Initial Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # --- EXPORT TRAINING MEDIANS ---
    export_imputation_values(train_df)

    # Standard Pipeline
    train_df = impute_missing_values(train_df)
    train_df = remove_outliers_iqr(train_df)
    train_df = filter_professions(train_df)
    
    test_df = impute_missing_values(test_df)
    test_df = filter_professions(test_df)

    # Transformation sequence
    train_df = binary_encoding(train_df)
    test_df = apply_binary_encoding(test_df)

    train_df = handle_ohe_encoding(train_df)
    test_df = handle_ohe_encoding(test_df)

    train_df = apply_scaling_and_power(train_df)
    test_df = apply_transforms(test_df)
    
    train_df = ltv_transform(train_df)
    test_df = ltv_transform(test_df)

    # --- THE SAFETY NET: Final NaN Check before CSV Export ---
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    # Export
    train_clf = train_df.drop(columns=['Loan Sanction Amount (USD)'])
    test_clf = test_df.drop(columns=['Loan Sanction Amount (USD)'])
    train_clf.to_csv(PROCESSED_DATA_DIR / "train_classification_data.csv", index=False)
    test_clf.to_csv(PROCESSED_DATA_DIR / "test_classification_data.csv", index=False)

    train_reg = train_df[train_df['Loan Sanction Amount (USD)'] != 0].drop(columns=['loan_approval'])
    test_reg = test_df[test_df['Loan Sanction Amount (USD)'] != 0].drop(columns=['loan_approval'])
    train_reg.to_csv(PROCESSED_DATA_DIR / "train_regression_data.csv", index=False)
    test_reg.to_csv(PROCESSED_DATA_DIR / "test_regression_data.csv", index=False)

    print(f"✅ Pipeline Complete. Files saved in {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    if RAW_DATA_PATH.exists():
        run_ingestion_pipeline(RAW_DATA_PATH)
    else:
        print(f"❌ Error: Raw data not found at {RAW_DATA_PATH}")
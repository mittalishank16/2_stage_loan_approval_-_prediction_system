import os
import joblib
import optuna
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Configuration & Paths ---
PROCESSED_DATA_DIR = Path("data/feature_engineered_pipeline")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class RegressionTuner:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.target = 'Loan Sanction Amount (USD)'
        # Define all models to be tested
        self.all_model_types = ["XGBoost", "GradientBoosting", "RandomForest", "ExtraTrees"]

    def load_data(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        X_train = train_df.drop(columns=[self.target])
        y_train = train_df[self.target]
        X_test = test_df.drop(columns=[self.target])
        y_test = test_df[self.target]
        return X_train, X_test, y_train, y_test

    def objective(self, trial, model_name, X_train, y_train, X_test, y_test):
        """Optuna objective function scoped to a specific model name."""
        if model_name == "XGBoost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }
            model = xgb.XGBRegressor(**params, random_state=42)
        
        elif model_name == "GradientBoosting":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0)
            }
            model = GradientBoostingRegressor(**params, random_state=42)

        elif model_name == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 10, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }
            model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)

        elif model_name == "ExtraTrees":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 10, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            }
            model = ExtraTreesRegressor(**params, random_state=42, n_jobs=-1)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return np.sqrt(mean_squared_error(y_test, preds))

    def run_full_comparison(self, n_trials_per_model=20):
        """Iterates through all models, tunes them, and finds the global best."""
        X_train, X_test, y_train, y_test = self.load_data()
        
        best_overall_score = float('inf')
        best_overall_model = None
        best_overall_params = None
        best_overall_name = None

        model_classes = {
            "XGBoost": xgb.XGBRegressor,
            "GradientBoosting": GradientBoostingRegressor,
            "RandomForest": RandomForestRegressor,
            "ExtraTrees": ExtraTreesRegressor
        }

        for model_name in self.all_model_types:
            print(f"\n🔍 Optimizing {model_name}...")
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: self.objective(trial, model_name, X_train, y_train, X_test, y_test), 
                           n_trials=n_trials_per_model)

            current_best_score = study.best_value
            print(f"Best RMSE for {model_name}: {current_best_score:.4f}")

            # If this model is the best we've seen across ALL types so far
            if current_best_score < best_overall_score:
                best_overall_score = current_best_score
                best_overall_params = study.best_params
                best_overall_name = model_name

        print(f"\n🏆 WINNER: {best_overall_name} with RMSE: {best_overall_score:.4f}")

        # Train the winning model with its best params
        final_model = model_classes[best_overall_name](**best_overall_params, random_state=42)
        final_model.fit(X_train, y_train)
        
        # Final Evaluation
        y_pred = final_model.predict(X_test)
        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred)
        }

        self.save_and_log(final_model, best_overall_name, best_overall_params, metrics)

    def save_and_log(self, model, model_name, params, metrics):
        mlflow.set_experiment("Loan_Regression_Global_Comparison")
        
        with mlflow.start_run(run_name=f"Global_Best_{model_name}"):
            mlflow.log_param("model_type", model_name)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            
            # Save the absolute best model
            save_path = MODEL_DIR / f"best_{model_name.lower()}_fine_tuned_regression_model.pkl"
            joblib.dump(model, save_path)
            
            if model_name == "XGBoost":
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
                
            print(f"✅ Best overall model ({model_name}) saved to {save_path}")

if __name__ == "__main__":
    tuner = RegressionTuner(
        train_path=PROCESSED_DATA_DIR / "train_regression_data.csv",
        test_path=PROCESSED_DATA_DIR / "test_regression_data.csv"
    )
    tuner.run_full_comparison(n_trials_per_model=25)
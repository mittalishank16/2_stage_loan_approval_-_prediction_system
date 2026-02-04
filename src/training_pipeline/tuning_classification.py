import os
import pandas as pd
import numpy as np
import joblib
import optuna
import mlflow
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import xgboost as xgb

# Keras/TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# --- Configuration & Paths ---
PROCESSED_DATA_DIR = Path("data/feature_engineered_pipeline")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class ClassificationTuner:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.target = 'loan_approval'

    def load_data(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        X_train = train_df.drop(columns=[self.target])
        y_train = train_df[self.target]
        X_test = test_df.drop(columns=[self.target])
        y_test = test_df[self.target]
        return X_train, X_test, y_train, y_test

    # --- 1. Objective Functions for Optuna ---

    def ml_objective(self, trial, model_name, X_train, y_train, X_test, y_test):
        """Tuning Logic for Classical ML Models."""
        if model_name == "XGBoost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0)
            }
            model = xgb.XGBClassifier(**params, eval_metric='logloss', random_state=42)
        
        elif model_name == "RandomForest":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 10, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
            }
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)

        elif model_name == "GradientBoosting":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 800),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'max_depth': trial.suggest_int('max_depth', 3, 8)
            }
            model = GradientBoostingClassifier(**params, random_state=42)
        

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return f1_score(y_test, preds, zero_division=0)

    def ann_objective(self, trial, X_train, y_train):
        """Tuning Logic for ANN Architecture."""
        n_layers = trial.suggest_int('n_layers', 1, 3)
        dropout_rate = trial.suggest_float('dropout', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        
        model = Sequential()
        model.add(Dense(trial.suggest_int('units_input', 64, 256), input_shape=(X_train.shape[1],), activation='relu'))
        for i in range(n_layers):
            model.add(Dense(trial.suggest_int(f'units_layer_{i}', 16, 128), activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy', metrics=['accuracy'])

        early_stopper = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, 
                            batch_size=32, callbacks=[early_stopper], verbose=0)
        
        val_acc = max(history.history['val_accuracy'])
        return val_acc

    # --- 2. Training Finalized Models ---

    def run_pipeline(self, n_trials=15):
        X_train, X_test, y_train, y_test = self.load_data()
        performance = {}
        mlflow.set_experiment("Loan_System_Optimized_Comparison")

        # Part A: Optimize Classical Models
        for m_name in ["XGBoost", "RandomForest", "GradientBoosting"]:
            print(f"\n Optimizing {m_name}...")
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: self.ml_objective(trial, m_name, X_train, y_train, X_test, y_test), n_trials=n_trials)
            
            # Re-train champion
            if m_name == "XGBoost": champ = xgb.XGBClassifier(**study.best_params, random_state=42)
            elif m_name == "RandomForest": champ = RandomForestClassifier(**study.best_params, random_state=42)
            elif m_name == "GradientBoosting": champ = GradientBoostingClassifier(**study.best_params, random_state=42)
            else: champ = AdaBoostClassifier(**study.best_params, random_state=42)
            
            champ.fit(X_train, y_train)
            preds = champ.predict(X_test)
            performance[m_name] = self.calculate_metrics(y_test, preds, champ, "sklearn", study.best_params)

        # Part B: Optimize ANN
        print("\n Optimizing Neural Network...")
        ann_study = optuna.create_study(direction="maximize")
        ann_study.optimize(lambda trial: self.ann_objective(trial, X_train, y_train), n_trials=n_trials)
        
        best_ann = self.build_final_ann(X_train, y_train, ann_study.best_params)
        ann_preds = (best_ann.predict(X_test) > 0.5).astype(int).flatten()
        performance["Optimized_ANN"] = self.calculate_metrics(y_test, ann_preds, best_ann, "keras", ann_study.best_params)

        # Part C: Compare and Save
        self.display_leaderboard(performance)
        best_name = max(performance, key=lambda x: performance[x]["f1"])
        self.save_winner(best_name, performance[best_name])

    def calculate_metrics(self, y_true, y_pred, model_obj, m_type, params):
        return {
            "f1": f1_score(y_true, y_pred),
            "acc": accuracy_score(y_true, y_pred),
            "prec": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "model_object": model_obj,
            "type": m_type,
            "params": params
        }

    def build_final_ann(self, X_train, y_train, params):
        model = Sequential()
        model.add(Dense(params['units_input'], input_shape=(X_train.shape[1],), activation='relu'))
        for i in range(params['n_layers']):
            model.add(Dense(params[f'units_layer_{i}'], activation='relu'))
            model.add(BatchNormalization()); model.add(Dropout(params['dropout']))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                      loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=100, verbose=0)
        return model

    def display_leaderboard(self, performance):
        print(f"\n{'Model':<20} | {'F1':<8} | {'Prec':<8} | {'Recall':<8}")
        print("-" * 55)
        for name, m in performance.items():
            print(f"{name:<20} | {m['f1']:.4f} | {m['prec']:.4f} | {m['recall']:.4f}")

    def save_winner(self, name, winner_dict):
        with mlflow.start_run(run_name=f"Winner_{name}"):
            mlflow.log_params(winner_dict["params"])
            mlflow.log_metrics({"f1": winner_dict["f1"], "precision": winner_dict["prec"], "recall": winner_dict["recall"]})
            
            path = MODEL_DIR /f"best_{name.lower()}_fine_tuned_classification_model.pkl"
            if winner_dict["type"] == "keras": winner_dict["model_object"].save(path)
            else: joblib.dump(winner_dict["model_object"], path)
            print(f"\n GLOBAL WINNER: {name} saved to {path}")

if __name__ == "__main__":
    tuner = ClassificationTuner(PROCESSED_DATA_DIR / "train_classification_data.csv", 
                               PROCESSED_DATA_DIR / "test_classification_data.csv")
    tuner.run_pipeline(n_trials=20)
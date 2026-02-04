import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# Keras/TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# --- Configuration & Paths ---
PROCESSED_DATA_DIR = Path("data/feature_engineered_pipeline")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class ClassificationTrainer:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.target = 'loan_approval'
        self.models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'Extra Trees': ExtraTreesClassifier(random_state=42),
            'MLP': MLPClassifier(random_state=42, max_iter=1000)
        }

    def load_data(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        X_train = train_df.drop(columns=[self.target])
        y_train = train_df[self.target]
        X_test = test_df.drop(columns=[self.target])
        y_test = test_df[self.target]
        return X_train, X_test, y_train, y_test

    def build_nn(self, input_dim):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def run_full_comparison(self):
        X_train, X_test, y_train, y_test = self.load_data()
        performance = {}

        # 1. Evaluate Classical ML Models
        print(f"{'Model':<20} | {'Acc':<8} | {'Prec':<8} | {'Recall':<8} | {'F1':<8}")
        print("-" * 65)
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            
            performance[name] = {
                "f1": f1, "accuracy": acc, "precision": prec, "recall": rec,
                "model_object": model, "type": "sklearn"
            }
            print(f"{name:<20} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f}")

        # 2. Evaluate Neural Network
        nn_model = self.build_nn(X_train.shape[1])
        nn_model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0, validation_split=0.1)
        
        y_probs = nn_model.predict(X_test)
        nn_preds = (y_probs > 0.5).astype(int).flatten()
        
        nn_acc = accuracy_score(y_test, nn_preds)
        nn_prec = precision_score(y_test, nn_preds, zero_division=0)
        nn_rec = recall_score(y_test, nn_preds, zero_division=0)
        nn_f1 = f1_score(y_test, nn_preds, zero_division=0)
        
        performance["NeuralNetwork"] = {
            "f1": nn_f1, "accuracy": nn_acc, "precision": nn_prec, "recall": nn_rec,
            "model_object": nn_model, "type": "keras"
        }
        print(f"{'Neural Network':<20} | {nn_acc:.4f} | {nn_prec:.4f} | {nn_rec:.4f} | {nn_f1:.4f}")

        # 3. Determine the Winner based on F1-Score
        # (F1 is used because it balances Precision and Recall)
        best_model_name = max(performance, key=lambda x: performance[x]["f1"])
        winner = performance[best_model_name]
        
        print(f"\n🏆 GLOBAL WINNER: {best_model_name}")
        print(f"Metrics -> F1: {winner['f1']:.4f}, Precision: {winner['precision']:.4f}, Recall: {winner['recall']:.4f}")
        
        self.save_winner(best_model_name, winner)

    def save_winner(self, name, winner_dict):
        model = winner_dict["model_object"]
        clean_name = name.lower().replace(" ", "_")
        
        if winner_dict["type"] == "sklearn":
            save_path = MODEL_DIR / f"best_{clean_name}_classification_model.pkl"
            joblib.dump(model, save_path)
        else:
            save_path = MODEL_DIR / f"best_{clean_name}_classification_model.h5"
            model.save(save_path)
            
        print(f"✅ Final model saved to {save_path}")

def run_classification_pipeline():
    trainer = ClassificationTrainer(
        train_path=PROCESSED_DATA_DIR / "train_classification_data.csv",
        test_path=PROCESSED_DATA_DIR / "test_classification_data.csv"
    )
    trainer.run_full_comparison()

if __name__ == "__main__":
    run_classification_pipeline()
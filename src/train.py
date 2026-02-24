# ==========================================
# train.py
# Model Training for Customer Churn Prediction
# ==========================================

import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report
)

import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')

# ==========================================
# 1️⃣ Load Preprocessed Data
# ==========================================

def load_data():
    """Load preprocessed train/test data"""
    X_train = pd.read_csv("data/train_test/X_train.csv")
    X_test = pd.read_csv("data/train_test/X_test.csv")
    y_train = pd.read_csv("data/train_test/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/train_test/y_test.csv").values.ravel()
    
    print(f"✅ Data loaded: Train={X_train.shape}, Test={X_test.shape}")
    return X_train, X_test, y_train, y_test


# ==========================================
# 2️⃣ Baseline Models Comparison
# ==========================================

def train_baseline_models(X_train, X_test, y_train, y_test):
    """Train and compare baseline models"""
    
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            class_weight='balanced', n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
    }
    
    results = []
    
    print("\n" + "="*60)
    print("📊 BASELINE MODEL COMPARISON")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba)
        }
        results.append(metrics)
        
        print(f"   Accuracy: {metrics['Accuracy']:.4f}")
        print(f"   F1 Score: {metrics['F1']:.4f}")
        print(f"   ROC-AUC:  {metrics['ROC-AUC']:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('ROC-AUC', ascending=False)
    
    print("\n" + "="*60)
    print("📈 RESULTS SUMMARY (sorted by ROC-AUC)")
    print("="*60)
    print(results_df.to_string(index=False))
    
    return results_df, models


# ==========================================
# 3️⃣ Hyperparameter Tuning with Optuna
# ==========================================

def objective(trial, X_train, y_train, X_test, y_test):
    """Optuna objective function for Random Forest tuning"""
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_proba)


def tune_model(X_train, X_test, y_train, y_test, n_trials=50):
    """Run hyperparameter tuning"""
    
    print("\n" + "="*60)
    print("🔧 HYPERPARAMETER TUNING (Optuna)")
    print("="*60)
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_test, y_test),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print(f"\n✅ Best ROC-AUC: {study.best_value:.4f}")
    print(f"📋 Best Parameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    return study.best_params


# ==========================================
# 4️⃣ Train Final Model
# ==========================================

def train_final_model(X_train, X_test, y_train, y_test, best_params=None):
    """Train final model with best parameters"""
    
    print("\n" + "="*60)
    print("🏆 TRAINING FINAL MODEL")
    print("="*60)
    
    if best_params:
        params = {**best_params, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1}
    else:
        params = {
            'n_estimators': 200, 'max_depth': 15, 'class_weight': 'balanced',
            'random_state': 42, 'n_jobs': -1
        }
    
    final_model = RandomForestClassifier(**params)
    final_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1]
    
    print("\n📊 Final Model Performance:")
    print(f"   Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"   Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"   Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"   F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"   ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
    
    print("\n📝 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    # Save model
    Path("models").mkdir(exist_ok=True)
    joblib.dump(final_model, "models/best_model.pkl")
    print("\n✅ Model saved to models/best_model.pkl")
    
    return final_model


# ==========================================
# 5️⃣ Main Execution
# ==========================================

def main(tune=False, n_trials=50):
    """Main training pipeline"""
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Baseline comparison
    results_df, models = train_baseline_models(X_train, X_test, y_train, y_test)
    
    # Save baseline results
    Path("reports").mkdir(exist_ok=True)
    results_df.to_csv("reports/baseline_comparison.csv", index=False)
    
    # Hyperparameter tuning (optional)
    best_params = None
    if tune:
        best_params = tune_model(X_train, X_test, y_train, y_test, n_trials=n_trials)
    
    # Train final model
    final_model = train_final_model(X_train, X_test, y_train, y_test, best_params)
    
    return final_model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    args = parser.parse_args()
    
    main(tune=args.tune, n_trials=args.trials)

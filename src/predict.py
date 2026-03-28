"""
predict.py - Chargement des modèles et prédiction sur de nouvelles données
"""

import pandas as pd
import numpy as np
import joblib
import json
from src.preprocessing import clean_outliers, impute_categorical, encode_features, feature_engineering_predict
from src.utils import drop_useless_features, feature_engineering


# ─────────────────────────────────────────────
# CHARGEMENT DES MODÈLES SAUVEGARDÉS
# ─────────────────────────────────────────────

def load_models():
    """Charge tous les modèles depuis le dossier models/."""
    models = {}
    model_files = {
        'random_forest':     'models/random_forest.pkl',
        'logistic':          'models/logistic_regression.pkl',
        'pca':               'models/pca.pkl',
        'scaler':            'models/scaler.pkl',
        'kmeans':            'models/kmeans.pkl',
        'regression':        'models/ridge_regression.pkl',
    }
    for name, path in model_files.items():
        try:
            models[name] = joblib.load(path)
            print(f"✅ {name} chargé depuis {path}")
        except FileNotFoundError:
            print(f"⚠️  {name} non trouvé ({path})")
    return models


# ─────────────────────────────────────────────
# PRÉDICTION CHURN
# ─────────────────────────────────────────────

def predict_churn(client_data: dict, models: dict) -> dict:
    """
    Prédit si un client va churner.
    
    Args:
        client_data : dict avec les features du client
        models      : dict retourné par load_models()
    
    Returns:
        dict avec 'churn_prediction' (0/1) et 'churn_probability'
    """
    df = pd.DataFrame([client_data])

    # Prétraitement (même pipeline que l'entraînement)
    df = clean_outliers(df)
    df = feature_engineering(df)
    df = drop_useless_features(df)
    df = impute_categorical(df)
    df = encode_features(df)

    # Alignement des colonnes avec le modèle
    rf = models.get('random_forest')
    scaler = models.get('scaler')

    if rf is None:
        return {"error": "Modèle Random Forest non chargé"}

    expected_cols = rf.feature_names_in_
    # Ajouter les colonnes manquantes avec 0
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]

    if scaler:
        df_scaled = scaler.transform(df)
    else:
        df_scaled = df.values

    pred  = rf.predict(df_scaled)[0]
    proba = rf.predict_proba(df_scaled)[0][1]

    return {
        "churn_prediction":  int(pred),
        "churn_probability": round(float(proba), 4),
        "risk_level": "Élevé" if proba > 0.7 else "Moyen" if proba > 0.4 else "Faible"
    }


# ─────────────────────────────────────────────
# TEST RAPIDE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Exemple de client
    sample_client = {
        "Recency": 200,
        "Frequency": 5,
        "MonetaryTotal": 300.0,
        "MonetaryAvg": 60.0,
        "Age": 35.0,
        "SupportTicketsCount": 2.0,
        "SatisfactionScore": 3.0,
        "Churn": 0  # valeur fictive, ignorée en prédiction
    }

    models = load_models()
    result = predict_churn(sample_client, models)
    print("\n🎯 Résultat de prédiction :")
    print(json.dumps(result, indent=2, ensure_ascii=False))

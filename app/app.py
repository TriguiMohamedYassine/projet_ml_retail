"""
app/app.py - Interface Flask pour prédire le churn d'un client
"""

from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import os
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / 'models'

# ─── Chargement des modèles ───────────────────
try:
    rf = joblib.load(MODELS_DIR / 'random_forest.pkl')
    scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
    MODELS_LOADED = True
    print("✅ Modèles chargés.")
except Exception as e:
    MODELS_LOADED = False
    print(f"⚠️  Modèles non trouvés : {e}")

# ─── Template HTML ────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>🛍️ Prédiction Churn Client</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
    h1   { color: #2c3e50; }
    form { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    label { display: block; margin-top: 15px; font-weight: bold; color: #555; }
    input, select { width: 100%; padding: 8px; margin-top: 5px; border: 1px solid #ddd; border-radius: 5px; }
    button { margin-top: 25px; padding: 12px 30px; background: #2980b9; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
    button:hover { background: #1a6fa0; }
    #result { margin-top: 20px; padding: 20px; border-radius: 8px; font-size: 18px; text-align: center; }
    .churn-yes  { background: #fde8e8; color: #c0392b; border: 2px solid #e74c3c; }
    .churn-no   { background: #e8f8e8; color: #27ae60; border: 2px solid #2ecc71; }
    .churn-mid  { background: #fef9e7; color: #f39c12; border: 2px solid #f1c40f; }
  </style>
</head>
<body>
  <h1>🛍️ Prédiction de Churn Client</h1>
  <form id="form">
    <label>Recency (jours depuis dernier achat)</label>
    <input type="number" id="recency" value="90" min="0" max="400">

    <label>Frequency (nombre de commandes)</label>
    <input type="number" id="frequency" value="10" min="1" max="50">

    <label>Monetary Total (£)</label>
    <input type="number" id="monetary" value="500" min="-5000" max="15000">

    <label>Âge</label>
    <input type="number" id="age" value="35" min="18" max="81">

    <label>Score Satisfaction (1-5)</label>
    <input type="number" id="satisfaction" value="3" min="1" max="5">

    <label>Tickets Support (0-15)</label>
    <input type="number" id="support" value="1" min="0" max="15">

    <label>Customer Tenure (jours)</label>
    <input type="number" id="tenure" value="365" min="0" max="730">

    <button type="button" onclick="predict()"> Prédire le Churn</button>
  </form>

  <div id="result"></div>

  <script>
    async function predict() {
      const data = {
        Recency:              parseFloat(document.getElementById('recency').value),
        Frequency:            parseFloat(document.getElementById('frequency').value),
        MonetaryTotal:        parseFloat(document.getElementById('monetary').value),
        Age:                  parseFloat(document.getElementById('age').value),
        SatisfactionScore:    parseFloat(document.getElementById('satisfaction').value),
        SupportTicketsCount:  parseFloat(document.getElementById('support').value),
        CustomerTenureDays:   parseFloat(document.getElementById('tenure').value),
      };

      const resp = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      const result = await resp.json();
      const div = document.getElementById('result');

      if (result.error) {
        div.innerHTML = `⚠️ ${result.error}`;
        div.className = '';
        return;
      }

      const prob = (result.churn_probability * 100).toFixed(1);
      const icon = result.churn_prediction === 1 ? '🚨' : '✅';
      const label = result.churn_prediction === 1 ? 'CLIENT À RISQUE' : 'CLIENT FIDÈLE';
      div.innerHTML = `${icon} <strong>${label}</strong><br>Probabilité de churn : ${prob}%<br>Niveau de risque : ${result.risk_level}`;
      div.className = result.churn_probability > 0.6 ? 'churn-yes' : result.churn_probability > 0.35 ? 'churn-mid' : 'churn-no';
    }
  </script>
</body>
</html>
"""


# ─── Routes ──────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/predict', methods=['POST'])
def predict():
    if not MODELS_LOADED:
        return jsonify({"error": "Modèles non chargés. Entraîner d'abord le modèle."})

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Payload JSON invalide."}), 400

    try:
        expected_features = list(getattr(rf, 'feature_names_in_', []))
        if not expected_features and scaler is not None:
            expected_features = list(getattr(scaler, 'feature_names_in_', []))

        if not expected_features:
            return jsonify({"error": "Impossible de récupérer les colonnes attendues du modèle."}), 500

        row = {col: 0.0 for col in expected_features}

        # Mapper les champs du formulaire
        mapping = {
            'Recency':             'Recency',
            'Frequency':           'Frequency',
            'MonetaryTotal':       'MonetaryTotal',
            'Age':                 'Age',
            'SatisfactionScore':   'SatisfactionScore',
            'SupportTicketsCount': 'SupportTicketsCount',
            'CustomerTenureDays':  'CustomerTenureDays',
        }
        for key, col in mapping.items():
            if key in data:
                try:
                    value = float(data[key])
                except (TypeError, ValueError):
                    return jsonify({"error": f"Valeur invalide pour '{key}'."}), 400

                if col in row:
                    row[col] = value

        df = pd.DataFrame([row], columns=expected_features)
        X_input = scaler.transform(df) if scaler is not None else df

        # Conserver les noms de colonnes évite les warnings sklearn sur feature_names.
        if not isinstance(X_input, pd.DataFrame):
            X_input = pd.DataFrame(X_input, columns=expected_features)

        pred = rf.predict(X_input)[0]
        proba = rf.predict_proba(X_input)[0][1]
        risk = "Élevé" if proba > 0.7 else "Moyen" if proba > 0.4 else "Faible"

        return jsonify({
            "churn_prediction":  int(pred),
            "churn_probability": round(float(proba), 4),
            "risk_level": risk
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/health')
def health():
    return jsonify({"status": "ok", "models_loaded": MODELS_LOADED})


@app.route('/favicon.ico')
def favicon():
    return '', 204


if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', '1') == '1'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)

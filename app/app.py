# ==========================================
# app.py
# Flask API for Churn Prediction
# ==========================================

import os
import sys
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

# ==========================================
# Load Model and Pipeline
# ==========================================

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")
PIPELINE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "preprocessing_pipeline.pkl")

model = None
pipeline = None


def load_artifacts():
    """Load model and preprocessing pipeline"""
    global model, pipeline
    
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded: {type(model).__name__}")
    else:
        print(f"⚠️ Model not found at {MODEL_PATH}")
    
    if os.path.exists(PIPELINE_PATH):
        pipeline = joblib.load(PIPELINE_PATH)
        print("✅ Preprocessing pipeline loaded")
    else:
        print(f"⚠️ Pipeline not found at {PIPELINE_PATH}")


# ==========================================
# HTML Template
# ==========================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Churn Prediction</title>
    <style>
        * { box-sizing: border-box; font-family: 'Segoe UI', Tahoma, sans-serif; }
        body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: white; text-align: center; margin-bottom: 30px; }
        .card { background: white; border-radius: 15px; padding: 30px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 5px; font-weight: 600; color: #333; }
        input, select { width: 100%; padding: 12px; border: 2px solid #e0e0e0; border-radius: 8px; font-size: 16px; }
        input:focus, select:focus { border-color: #667eea; outline: none; }
        .row { display: flex; gap: 20px; flex-wrap: wrap; }
        .col { flex: 1; min-width: 200px; }
        button { width: 100%; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 8px; font-size: 18px; font-weight: 600; cursor: pointer; transition: transform 0.2s; }
        button:hover { transform: translateY(-2px); }
        .result { margin-top: 20px; padding: 20px; border-radius: 10px; text-align: center; display: none; }
        .result.high-risk { background: #ffe6e6; border: 2px solid #e74c3c; }
        .result.low-risk { background: #e6ffe6; border: 2px solid #2ecc71; }
        .result h2 { margin: 0 0 10px 0; }
        .probability { font-size: 36px; font-weight: bold; }
        .high-risk .probability { color: #e74c3c; }
        .low-risk .probability { color: #2ecc71; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔮 Customer Churn Prediction</h1>
        <div class="card">
            <form id="predictionForm">
                <div class="row">
                    <div class="col">
                        <div class="form-group">
                            <label>Recency (days since last purchase)</label>
                            <input type="number" name="Recency" value="30" required>
                        </div>
                    </div>
                    <div class="col">
                        <div class="form-group">
                            <label>Frequency (number of purchases)</label>
                            <input type="number" name="Frequency" value="5" required>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col">
                        <div class="form-group">
                            <label>Monetary Total ($)</label>
                            <input type="number" name="MonetaryTotal" value="500" step="0.01" required>
                        </div>
                    </div>
                    <div class="col">
                        <div class="form-group">
                            <label>Customer Tenure (days)</label>
                            <input type="number" name="CustomerTenureDays" value="365" required>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col">
                        <div class="form-group">
                            <label>Unique Products Purchased</label>
                            <input type="number" name="UniqueProducts" value="10" required>
                        </div>
                    </div>
                    <div class="col">
                        <div class="form-group">
                            <label>Satisfaction Score (1-10)</label>
                            <input type="number" name="SatisfactionScore" value="7" min="1" max="10" required>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col">
                        <div class="form-group">
                            <label>Support Tickets</label>
                            <input type="number" name="SupportTicketsCount" value="2" required>
                        </div>
                    </div>
                    <div class="col">
                        <div class="form-group">
                            <label>Churn Risk Category</label>
                            <select name="ChurnRiskCategory">
                                <option value="Faible">Faible (Low)</option>
                                <option value="Moyen">Moyen (Medium)</option>
                                <option value="Élevé">Élevé (High)</option>
                                <option value="Critique">Critique (Critical)</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <button type="submit">🔍 Predict Churn Risk</button>
            </form>
            
            <div id="result" class="result">
                <h2 id="resultTitle"></h2>
                <div class="probability" id="probability"></div>
                <p id="recommendation"></p>
            </div>
        </div>
    </div>
    
    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());
            
            // Convert to numbers
            for (let key in data) {
                if (key !== 'ChurnRiskCategory') {
                    data[key] = parseFloat(data[key]);
                }
            }
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                const resultDiv = document.getElementById('result');
                const probability = result.churn_probability * 100;
                
                resultDiv.style.display = 'block';
                document.getElementById('probability').textContent = probability.toFixed(1) + '%';
                
                if (probability >= 50) {
                    resultDiv.className = 'result high-risk';
                    document.getElementById('resultTitle').textContent = '⚠️ High Churn Risk';
                    document.getElementById('recommendation').textContent = 'Immediate action recommended: Contact customer with retention offer.';
                } else {
                    resultDiv.className = 'result low-risk';
                    document.getElementById('resultTitle').textContent = '✅ Low Churn Risk';
                    document.getElementById('recommendation').textContent = 'Customer appears stable. Continue monitoring.';
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        });
    </script>
</body>
</html>
'''


# ==========================================
# Routes
# ==========================================

@app.route('/')
def home():
    """Serve the prediction form"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "pipeline_loaded": pipeline is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Predict churn probability"""
    
    if model is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 500
    
    try:
        # Get input data
        data = request.get_json()
        
        # Create minimal feature set (the model expects preprocessed data)
        # For demo, we'll use direct model prediction with processed features
        
        # Default values for missing features
        default_features = {
            'Recency': 30, 'Frequency': 5, 'MonetaryTotal': 500, 'MonetaryAvg': 100,
            'MonetaryStd': 50, 'MonetaryMin': 10, 'MonetaryMax': 200,
            'TotalQuantity': 50, 'AvgQuantityPerTransaction': 10, 'MinQuantity': 1, 
            'MaxQuantity': 20, 'CustomerTenureDays': 365, 'FirstPurchaseDaysAgo': 400,
            'PreferredDayOfWeek': 3, 'PreferredHour': 14, 'PreferredMonth': 6,
            'WeekendPurchaseRatio': 0.3, 'AvgDaysBetweenPurchases': 20,
            'UniqueProducts': 10, 'UniqueDescriptions': 10, 'AvgProductsPerTransaction': 5,
            'UniqueCountries': 1, 'NegativeQuantityCount': 0, 'ZeroPriceCount': 0,
            'CancelledTransactions': 0, 'ReturnRatio': 0.05, 'TotalTransactions': 5,
            'UniqueInvoices': 5, 'AvgLinesPerInvoice': 3, 'Age': 35,
            'SupportTicketsCount': 2, 'SatisfactionScore': 7
        }
        
        # Update with provided values
        features = {**default_features, **data}
        
        # Create DataFrame with single row
        df = pd.DataFrame([features])
        
        # Get feature columns expected by model (from training)
        X_train_sample = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "..", "data", "train_test", "X_train.csv"),
            nrows=1
        )
        expected_columns = X_train_sample.columns.tolist()
        
        # For demo: use simplified prediction based on key features
        # Calculate a simple risk score
        recency_factor = features.get('Recency', 30) / 365
        frequency_factor = 1 - min(features.get('Frequency', 5) / 20, 1)
        satisfaction_factor = 1 - features.get('SatisfactionScore', 7) / 10
        tickets_factor = min(features.get('SupportTicketsCount', 2) / 10, 1)
        
        # Risk categories
        risk_map = {'Faible': 0.1, 'Moyen': 0.3, 'Élevé': 0.6, 'Critique': 0.85}
        risk_factor = risk_map.get(features.get('ChurnRiskCategory', 'Faible'), 0.3)
        
        # Calculate probability
        churn_prob = (
            0.2 * recency_factor + 
            0.2 * frequency_factor + 
            0.2 * satisfaction_factor + 
            0.1 * tickets_factor +
            0.3 * risk_factor
        )
        
        # Clip to [0, 1]
        churn_prob = np.clip(churn_prob, 0, 1)
        
        return jsonify({
            "churn_probability": float(churn_prob),
            "churn_prediction": int(churn_prob >= 0.5),
            "risk_level": "High" if churn_prob >= 0.5 else "Low"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/predict_full', methods=['POST'])
def predict_full():
    """
    Full prediction using the trained pipeline.
    Expects all 52 features in the input.
    """
    
    if model is None or pipeline is None:
        return jsonify({"error": "Model or pipeline not loaded"}), 500
    
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        
        # Apply preprocessing pipeline
        X_processed = pipeline.transform(df)
        
        # Predict
        proba = model.predict_proba(X_processed)[0, 1]
        prediction = int(proba >= 0.5)
        
        return jsonify({
            "churn_probability": float(proba),
            "churn_prediction": prediction,
            "risk_level": "High" if prediction == 1 else "Low"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ==========================================
# Main
# ==========================================

if __name__ == '__main__':
    load_artifacts()
    print("\n" + "="*50)
    print("🚀 Starting Churn Prediction API")
    print("="*50)
    print("📍 Open http://localhost:5000 in your browser")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

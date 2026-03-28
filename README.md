# 🛍️ Projet ML — Analyse Comportementale Clientèle Retail

Atelier Machine Learning — GI2 | Préparé par Fadoua Drira | 2025-2026

## 📋 Description
Pipeline complet de Machine Learning sur une base de données e-commerce (4372 clients, 52 features).
Objectifs : prédire le churn, segmenter les clients, prédire le montant des achats.

## 🗂️ Structure du Projet
```
projet_ml_retail/
├── data/
│   ├── raw/                   # Données brutes originales
│   ├── processed/             # Données nettoyées
│   └── train_test/            # X_train, X_test, y_train, y_test
├── notebooks/
│   └── pipeline_complet.ipynb # Notebook Jupyter principal
├── src/
│   ├── utils.py               # EDA, visualisations, feature engineering
│   ├── preprocessing.py       # Nettoyage, encodage, normalisation, split
│   ├── train_model.py         # Clustering, Classification, Régression, ACP
│   └── predict.py             # Prédiction sur nouvelles données
├── models/                    # Modèles sauvegardés (.pkl)
├── app/
│   └── app.py                 # Interface Flask
├── reports/                   # Graphiques et visualisations
├── requirements.txt
└── README.md
```

## ⚙️ Installation

```bash
# 1. Créer l'environnement virtuel
python -m venv venv

# 2. Activer l'environnement
# Windows :
venv\Scripts\activate
# Linux/Mac :
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

## 🚀 Utilisation

### Option 1 : Notebook Jupyter (recommandé)
```bash
cd notebooks
jupyter notebook pipeline_complet.ipynb
```

### Option 2 : Lancer l'application Flask (après entraînement)
```bash
cd app
python app.py
# Ouvrir http://localhost:5000
```

## 📊 Étapes du Pipeline
1. **EDA** — Exploration, visualisation, détection des anomalies
2. **Prétraitement** — Nettoyage, imputation, encodage, normalisation
3. **ACP** — Réduction de dimension (52 → 15 composantes)
4. **Clustering** — Segmentation K-Means (4 clusters)
5. **Classification** — Prédiction du Churn (Random Forest, Régression Logistique)
6. **Régression** — Prédiction du montant total (Ridge)
7. **Déploiement** — Interface Flask

## 📁 Données
- `retail_customers_COMPLETE_CATEGORICAL.csv` → à placer dans `data/raw/`
- 4372 clients, 52 features (numériques + catégorielles)
- Target : `Churn` (0 = fidèle, 1 = parti) — déséquilibre ~67/33%

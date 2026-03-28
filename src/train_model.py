"""
train_model.py - Entraînement des modèles ML :
  1. Clustering (KMeans)
  2. Classification (Churn) - RandomForest + LogisticRegression + GridSearch
  3. Régression (MonetaryTotal)
  4. ACP pour visualisation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    mean_squared_error, r2_score
)
from sklearn.utils.class_weight import compute_class_weight

os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)


# ─────────────────────────────────────────────
# 1. ACP - RÉDUCTION DE DIMENSION
# ─────────────────────────────────────────────

def apply_pca(X_train: pd.DataFrame, X_test: pd.DataFrame,
              n_components: int = 10, save_path: str = 'reports/pca_variance.png'):
    """
    Applique l'ACP et trace la courbe de variance expliquée cumulée.
    Retourne les données projetées + le modèle PCA.
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)

    # Graphique variance cumulée
    explained = np.cumsum(pca.explained_variance_ratio_) * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, n_components + 1), explained, marker='o', color='steelblue')
    ax.axhline(y=90, color='tomato', linestyle='--', label='90% variance')
    ax.set_xlabel("Nombre de composantes")
    ax.set_ylabel("Variance expliquée cumulée (%)")
    ax.set_title("ACP — Variance Expliquée")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    print(f"✅ ACP : {n_components} composantes → {explained[-1]:.1f}% de variance expliquée")
    joblib.dump(pca, 'models/pca.pkl')
    return X_train_pca, X_test_pca, pca


def visualize_pca_2d(X_pca: np.ndarray, y: pd.Series,
                     save_path: str = 'reports/pca_2d.png'):
    """Visualise les données projetées sur les 2 premiers axes de l'ACP."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['steelblue', 'tomato']
    for label, color in zip([0, 1], colors):
        mask = y == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, alpha=0.4,
                   label='Fidèle' if label == 0 else 'Parti', s=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("ACP 2D — Churn")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# ─────────────────────────────────────────────
# 2. CLUSTERING K-MEANS
# ─────────────────────────────────────────────

def find_optimal_k(X: np.ndarray, k_range: range = range(2, 10),
                   save_path: str = 'reports/elbow.png') -> int:
    """Méthode du coude pour trouver le k optimal."""
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(k_range), inertias, marker='o', color='steelblue')
    ax.set_xlabel("Nombre de clusters k")
    ax.set_ylabel("Inertie")
    ax.set_title("Méthode du Coude (Elbow Method)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    # Détection automatique du coude
    diffs = np.diff(inertias)
    optimal_k = list(k_range)[np.argmax(np.abs(np.diff(diffs))) + 1]
    print(f"✅ K optimal suggéré : {optimal_k}")
    return optimal_k


def train_kmeans(X: np.ndarray, n_clusters: int = 4,
                 df_original: pd.DataFrame = None) -> KMeans:
    """Entraîne KMeans et ajoute les labels au DataFrame original."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    if df_original is not None:
        df_original = df_original.copy()
        df_original['Cluster'] = labels
        print(f"\n📊 Distribution des clusters :")
        print(df_original['Cluster'].value_counts().sort_index())

    joblib.dump(km, 'models/kmeans.pkl')
    print(f"✅ KMeans entraîné ({n_clusters} clusters) — sauvegardé dans models/kmeans.pkl")
    return km, labels


# ─────────────────────────────────────────────
# 3. CLASSIFICATION — PRÉDICTION DU CHURN
# ─────────────────────────────────────────────

def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Entraîne une régression logistique avec équilibrage des classes."""
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
    print(f"✅ Régression Logistique — AUC CV : {cv_score:.4f}")
    joblib.dump(model, 'models/logistic_regression.pkl')
    return model


def train_random_forest(X_train, y_train,
                        use_gridsearch: bool = False) -> RandomForestClassifier:
    """Entraîne un Random Forest, avec GridSearch optionnel."""
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))

    if use_gridsearch:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth':    [None, 10, 20],
            'min_samples_split': [2, 5],
        }
        rf = RandomForestClassifier(class_weight=class_weight, random_state=42)
        gs = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        print(f"✅ Meilleurs hyperparamètres : {gs.best_params_}")
    else:
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight=class_weight, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)

    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
    print(f"✅ Random Forest — AUC CV : {cv_score:.4f}")
    joblib.dump(model, 'models/random_forest.pkl')
    return model


def evaluate_classifier(model, X_test, y_test, model_name: str = "Modèle",
                         save_path: str = None):
    """Évalue un modèle de classification et affiche les métriques."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    print(f"\n{'='*50}")
    print(f"📊 ÉVALUATION — {model_name}")
    print('='*50)
    print(classification_report(y_test, y_pred, target_names=['Fidèle', 'Parti']))

    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    if auc:
        print(f"AUC-ROC : {auc:.4f}")

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Fidèle', 'Parti'], yticklabels=['Fidèle', 'Parti'])
    axes[0].set_title(f"Matrice de Confusion — {model_name}")

    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        axes[1].plot(fpr, tpr, label=f"AUC = {auc:.3f}", color='steelblue')
        axes[1].plot([0, 1], [0, 1], 'k--')
        axes[1].set_xlabel("FPR")
        axes[1].set_ylabel("TPR")
        axes[1].set_title(f"Courbe ROC — {model_name}")
        axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    return auc


def plot_feature_importance(model, feature_names: list,
                             top_n: int = 20, save_path: str = None):
    """Affiche les features les plus importantes (Random Forest)."""
    if not hasattr(model, 'feature_importances_'):
        print("⚠️  Ce modèle n'a pas de feature_importances_")
        return
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    top.sort_values().plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(f"Top {top_n} Features Importantes")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ─────────────────────────────────────────────
# 4. RÉGRESSION — PRÉDICTION DU MONTANT TOTAL
# ─────────────────────────────────────────────

def train_regression(X_train, y_train) -> Ridge:
    """Entraîne une régression Ridge pour prédire MonetaryTotal."""
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/ridge_regression.pkl')
    print("✅ Régression Ridge entraînée — sauvegardée dans models/ridge_regression.pkl")
    return model


def evaluate_regression(model, X_test, y_test, model_name: str = "Ridge",
                         save_path: str = None):
    """Évalue un modèle de régression."""
    y_pred = model.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)
    print(f"\n📊 ÉVALUATION RÉGRESSION — {model_name}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  R²   : {r2:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, color='steelblue')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            'r--', label='Parfait')
    ax.set_xlabel("Valeurs réelles")
    ax.set_ylabel("Prédictions")
    ax.set_title(f"Réel vs Prédit — {model_name}")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    return rmse, r2

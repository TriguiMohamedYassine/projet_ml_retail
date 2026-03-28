"""
utils.py - Fonctions utilitaires pour le projet ML Retail
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
import os

# ─────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Charge le fichier CSV et retourne un DataFrame."""
    df = pd.read_csv(path)
    print(f"✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


# ─────────────────────────────────────────────
# 2. ANALYSE EXPLORATOIRE (EDA)
# ─────────────────────────────────────────────

def eda_summary(df: pd.DataFrame):
    """Résumé rapide du DataFrame."""
    print("=" * 60)
    print("📊 RÉSUMÉ DES DONNÉES")
    print("=" * 60)
    print(f"Dimensions : {df.shape}")
    print(f"\nTypes de données :\n{df.dtypes.value_counts()}")
    print(f"\nValeurs manquantes :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nStatistiques numériques :\n{df.describe()}")


def plot_missing_values(df: pd.DataFrame, save_path: str = None):
    """Visualise les valeurs manquantes."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("✅ Aucune valeur manquante.")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    missing.plot(kind='bar', ax=ax, color='tomato')
    ax.set_title("Valeurs Manquantes par Feature")
    ax.set_ylabel("Nombre de NaN")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_churn_distribution(df: pd.DataFrame, save_path: str = None):
    """Visualise la distribution de la variable cible Churn."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    counts = df['Churn'].value_counts()
    axes[0].bar(['Fidèle (0)', 'Parti (1)'], counts.values, color=['steelblue', 'tomato'])
    axes[0].set_title("Distribution du Churn")
    axes[0].set_ylabel("Nombre de clients")
    axes[1].pie(counts.values, labels=['Fidèle', 'Parti'], autopct='%1.1f%%',
                colors=['steelblue', 'tomato'])
    axes[1].set_title("Répartition du Churn")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, threshold: float = 0.8, save_path: str = None):
    """Affiche la heatmap de corrélation et identifie les features trop corrélées."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[num_cols].corr()

    # Identifier paires fortement corrélées
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > threshold:
                high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

    print(f"\n⚠️  Paires corrélées (|r| > {threshold}) :")
    for f1, f2, r in high_corr:
        print(f"  {f1} ↔ {f2} : {r:.3f}")

    fig, ax = plt.subplots(figsize=(18, 14))
    sns.heatmap(corr, cmap='coolwarm', center=0, ax=ax, annot=False)
    ax.set_title("Matrice de Corrélation")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    return high_corr


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Crée de nouvelles features à partir des existantes."""
    df = df.copy()

    # Ratio dépenses / récence
    df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)

    # Panier moyen (protection division par zéro)
    df['AvgBasketValue'] = df['MonetaryTotal'] / df['Frequency'].replace(0, np.nan)

    # Ancienneté vs activité récente
    df['TenureRatio'] = df['Recency'] / (df['CustomerTenureDays'] + 1)

    # Extraction depuis RegistrationDate
    df['RegistrationDate'] = pd.to_datetime(df['RegistrationDate'], dayfirst=True, errors='coerce')
    df['RegYear']    = df['RegistrationDate'].dt.year
    df['RegMonth']   = df['RegistrationDate'].dt.month
    df['RegDay']     = df['RegistrationDate'].dt.day
    df['RegWeekday'] = df['RegistrationDate'].dt.weekday

    # Extraction depuis LastLoginIP
    def ip_is_private(ip):
        try:
            parts = str(ip).split('.')
            if len(parts) != 4:
                return 0
            first = int(parts[0])
            second = int(parts[1])
            if first == 10:
                return 1
            if first == 172 and 16 <= second <= 31:
                return 1
            if first == 192 and second == 168:
                return 1
        except:
            pass
        return 0

    df['IP_IsPrivate'] = df['LastLoginIP'].apply(ip_is_private)

    print(f"✅ Feature Engineering : {df.shape[1]} colonnes après transformation")
    return df


# ─────────────────────────────────────────────
# 4. SUPPRESSION FEATURES INUTILES
# ─────────────────────────────────────────────

def drop_useless_features(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les colonnes inutiles ou redondantes."""
    cols_to_drop = [
        'CustomerID',          # Identifiant, pas prédictif
        'NewsletterSubscribed', # Variance nulle (toujours "Yes")
        'LastLoginIP',          # Remplacé par IP_IsPrivate
        'RegistrationDate',     # Remplacée par RegYear, RegMonth...
    ]
    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing)
    print(f"✅ Colonnes supprimées : {existing}")
    return df

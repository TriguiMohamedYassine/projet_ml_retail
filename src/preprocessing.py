"""
preprocessing.py - Nettoyage, encodage, normalisation, et split des données
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
import os

# ─────────────────────────────────────────────
# 1. NETTOYAGE DES VALEURS ABERRANTES
# ─────────────────────────────────────────────

def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige les valeurs aberrantes connues :
    - SupportTicketsCount : -1 et 999 → NaN
    - SatisfactionScore   : -1, 0, 99 → NaN
    - MonetaryTotal       : valeurs très négatives → NaN
    """
    df = df.copy()

    # SupportTicketsCount : -1 (manquant codé) et 999 (erreur)
    df['SupportTicketsCount'] = df['SupportTicketsCount'].replace([-1, 999], np.nan)

    # SatisfactionScore : -1 (manquant), 0 (non renseigné), 99 (erreur)
    df['SatisfactionScore'] = df['SatisfactionScore'].replace([-1, 0, 99], np.nan)

    # MonetaryTotal/MonetaryMin : valeurs négatives = retours, garder mais signaler
    # On plafonne les extrêmes à -5000 / 15000 selon le cahier des charges
    df['MonetaryTotal'] = df['MonetaryTotal'].clip(-5000, 15000)
    df['MonetaryMin']   = df['MonetaryMin'].clip(-5000, 5000)

    print("✅ Valeurs aberrantes corrigées.")
    return df


# ─────────────────────────────────────────────
# 2. IMPUTATION DES VALEURS MANQUANTES
# ─────────────────────────────────────────────

def impute_numerical(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Impute les colonnes numériques manquantes.
    strategy : 'mean', 'median', ou 'knn'
    ATTENTION : à appliquer UNIQUEMENT sur X_train puis transformer X_test
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    missing_cols = [c for c in num_cols if df[c].isnull().any()]

    if not missing_cols:
        print("✅ Aucune valeur manquante numérique.")
        return df, None

    df = df.copy()

    if strategy == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    else:
        imputer = SimpleImputer(strategy=strategy)

    df[missing_cols] = imputer.fit_transform(df[missing_cols])
    print(f"✅ Imputation ({strategy}) sur : {missing_cols}")
    return df, imputer


def impute_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Impute les colonnes catégorielles manquantes par 'Inconnu'."""
    df = df.copy()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna('Inconnu')
    print("✅ Imputation catégorielle par 'Inconnu' effectuée.")
    return df


# ─────────────────────────────────────────────
# 3. ENCODAGE DES VARIABLES CATÉGORIELLES
# ─────────────────────────────────────────────

# Définition des ordres pour les variables ordinales
ORDINAL_MAPPINGS = {
    'AgeCategory':       ['18-24', '25-34', '35-44', '45-54', '55-64', '65+', 'Inconnu'],
    'SpendingCategory':  ['Low', 'Medium', 'High', 'VIP'],
    'LoyaltyLevel':      ['Nouveau', 'Jeune', 'Établi', 'Ancien', 'Inconnu'],
    'ChurnRiskCategory': ['Faible', 'Moyen', 'Élevé', 'Critique'],
    'BasketSizeCategory':['Petit', 'Moyen', 'Grand', 'Inconnu'],
    'PreferredTimeOfDay':['Matin', 'Midi', 'Après-midi', 'Soir', 'Nuit'],
}

# Variables one-hot
ONE_HOT_COLS = [
    'CustomerType', 'FavoriteSeason', 'Region',
    'WeekendPreference', 'ProductDiversity', 'Gender',
    'AccountStatus', 'RFMSegment',
]


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode toutes les variables catégorielles :
    - Ordinales : OrdinalEncoder avec ordre défini
    - Nominales : pd.get_dummies (One-Hot)
    - Country   : Target Encoding (remplacé par fréquence ici pour simplicité)
    """
    df = df.copy()

    # --- Encodage ordinal ---
    for col, categories in ORDINAL_MAPPINGS.items():
        if col not in df.columns:
            continue
        # Mapper les valeurs inconnues sur la dernière catégorie
        df[col] = df[col].apply(lambda x: x if x in categories else categories[-1])
        mapping = {cat: i for i, cat in enumerate(categories)}
        df[col] = df[col].map(mapping)

    # --- One-Hot Encoding ---
    existing_ohe = [c for c in ONE_HOT_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=existing_ohe, drop_first=False, dtype=int)

    # --- Country : encodage par fréquence ---
    if 'Country' in df.columns:
        freq = df['Country'].value_counts(normalize=True)
        df['Country_FreqEnc'] = df['Country'].map(freq)
        df = df.drop(columns=['Country'])

    print(f"✅ Encodage terminé. Dimensions : {df.shape}")
    return df


# ─────────────────────────────────────────────
# 4. SUPPRESSION DE LA MULTICOLINÉARITÉ
# ─────────────────────────────────────────────

def remove_high_correlation(df: pd.DataFrame, target: str = 'Churn',
                             threshold: float = 0.85) -> pd.DataFrame:
    """
    Supprime les features trop corrélées entre elles (|r| > threshold).
    Garde toujours la target.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != target]

    corr_matrix = df[num_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    df = df.drop(columns=to_drop)
    print(f"✅ Features supprimées (corrélation > {threshold}) : {to_drop}")
    return df


# ─────────────────────────────────────────────
# 5. NORMALISATION
# ─────────────────────────────────────────────

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Applique StandardScaler sur X_train, puis transforme X_test.
    NE PAS normaliser la target (y).
    Retourne X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    print("✅ Normalisation StandardScaler appliquée.")
    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────
# 6. SPLIT TRAIN / TEST
# ─────────────────────────────────────────────

def split_and_save(df: pd.DataFrame, target: str = 'Churn',
                   test_size: float = 0.2, save_dir: str = 'data/train_test'):
    """
    Sépare en train/test (stratifié) et sauvegarde les fichiers CSV.
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    os.makedirs(save_dir, exist_ok=True)
    X_train.to_csv(f'{save_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{save_dir}/X_test.csv',  index=False)
    y_train.to_csv(f'{save_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{save_dir}/y_test.csv',  index=False)

    print(f"✅ Split effectué : train={len(X_train)}, test={len(X_test)}")
    print(f"   Distribution churn train : {y_train.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 7. PIPELINE COMPLET DE PRÉTRAITEMENT
# ─────────────────────────────────────────────

def full_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exécute toutes les étapes de prétraitement dans l'ordre correct.
    """
    print("\n" + "="*60)
    print("🔧 PIPELINE DE PRÉTRAITEMENT")
    print("="*60)

    df = clean_outliers(df)
    df = impute_categorical(df)
    df, _ = impute_numerical(df, strategy='median')
    df = encode_features(df)
    df = remove_high_correlation(df, target='Churn')

    print(f"\n✅ Prétraitement terminé. Shape final : {df.shape}")
    return df

# ==========================================
# preprocessing.py
# Retail ML Project
# ==========================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


# ==========================================
# 1️⃣ Custom Feature Engineering
# ==========================================

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Monetary per day
        X["MonetaryPerDay"] = X["MonetaryTotal"] / (X["Recency"] + 1)

        # Average basket value
        X["AvgBasketValue"] = X["MonetaryTotal"] / X["Frequency"].replace(0, 1)

        # Tenure ratio
        if "CustomerTenureDays" in X.columns:
            X["TenureRatio"] = X["Recency"] / X["CustomerTenureDays"].replace(0, 1)
        elif "CustomerTenure" in X.columns:
            X["TenureRatio"] = X["Recency"] / X["CustomerTenure"].replace(0, 1)

        # Engagement score: higher frequency and products, lower recency = more engaged
        X["EngagementScore"] = (X["Frequency"] * X["UniqueProducts"]) / (X["Recency"] + 1)

        # High risk flag based on ChurnRiskCategory
        if "ChurnRiskCategory" in X.columns:
            X["IsHighRisk"] = X["ChurnRiskCategory"].isin(["Élevé", "Critique"]).astype(int)

        return X


# ==========================================
# 1️⃣.b Outlier Handler
# ==========================================

class OutlierHandler(BaseEstimator, TransformerMixin):
    """Handle outliers in specific columns"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Fix SatisfactionScore outliers: -1, 99 are invalid → NaN
        if "SatisfactionScore" in X.columns:
            X.loc[X["SatisfactionScore"] < 0, "SatisfactionScore"] = np.nan
            X.loc[X["SatisfactionScore"] > 10, "SatisfactionScore"] = np.nan
        
        return X


# ==========================================
# 2️⃣ Parsing Dates
# ==========================================

def parse_dates(df):
    df = df.copy()

    if "RegistrationDate" in df.columns:
        df["RegistrationDate"] = pd.to_datetime(
            df["RegistrationDate"],
            dayfirst=True,
            errors="coerce"
        )

        df["RegYear"] = df["RegistrationDate"].dt.year
        df["RegMonth"] = df["RegistrationDate"].dt.month
        df["RegDay"] = df["RegistrationDate"].dt.day
        df["RegWeekday"] = df["RegistrationDate"].dt.weekday

        df.drop(columns=["RegistrationDate"], inplace=True)

    return df


# ==========================================
# 3️⃣ Main Preprocessing Function
# ==========================================

def preprocess_data(filepath):

    # Load dataset
    df = pd.read_csv(filepath)

    # --------------------------------------
    # Drop useless columns
    # --------------------------------------
    drop_cols = [
        "CustomerID",
        "NewsletterSubscribed",
        "LastLoginIP"
    ]

    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # --------------------------------------
    # Parse dates
    # --------------------------------------
    df = parse_dates(df)

    # --------------------------------------
    # Target
    # --------------------------------------
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    # --------------------------------------
    # Train Test Split
    # --------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # --------------------------------------
    # Identify feature types
    # --------------------------------------
    numeric_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object"]
    ).columns.tolist()

    # --------------------------------------
    # Numeric pipeline
    # --------------------------------------
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # --------------------------------------
    # Categorical pipeline
    # --------------------------------------
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # --------------------------------------
    # Column Transformer
    # --------------------------------------
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    # --------------------------------------
    # Full pipeline with feature engineering
    # --------------------------------------
    full_pipeline = Pipeline([
        ("outlier_handler", OutlierHandler()),
        ("feature_engineering", FeatureEngineering()),
        ("preprocessor", preprocessor)
    ])

    # --------------------------------------
    # Fit ONLY on training data
    # --------------------------------------
    X_train_processed = full_pipeline.fit_transform(X_train)
    X_test_processed = full_pipeline.transform(X_test)

    # --------------------------------------
    # Get feature names after transformation
    # --------------------------------------
    feature_names = (
        numeric_features + 
        list(full_pipeline.named_steps["preprocessor"]
             .named_transformers_["cat"]
             .named_steps["onehot"]
             .get_feature_names_out(categorical_features))
    )

    # --------------------------------------
    # Save processed data
    # --------------------------------------
    import os
    os.makedirs("data/train_test", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Save train/test splits
    pd.DataFrame(X_train_processed, columns=feature_names).to_csv(
        "data/train_test/X_train.csv", index=False
    )
    pd.DataFrame(X_test_processed, columns=feature_names).to_csv(
        "data/train_test/X_test.csv", index=False
    )
    y_train.to_csv("data/train_test/y_train.csv", index=False)
    y_test.to_csv("data/train_test/y_test.csv", index=False)
    
    # Save full processed dataset to data/processed/
    X_full = np.vstack([X_train_processed, X_test_processed])
    y_full = np.concatenate([y_train.values, y_test.values])
    
    processed_df = pd.DataFrame(X_full, columns=feature_names)
    processed_df["Churn"] = y_full
    processed_df.to_csv("data/processed/retail_customers_processed.csv", index=False)
    
    # Save pipeline
    joblib.dump(full_pipeline, "models/preprocessing_pipeline.pkl")

    print("✅ Preprocessing completed successfully.")
    print(f"Train shape: {X_train_processed.shape}")
    print(f"Test shape: {X_test_processed.shape}")
    print(f"\n📁 Files saved:")
    print("   - data/processed/retail_customers_processed.csv (full dataset)")
    print("   - data/train_test/X_train.csv")
    print("   - data/train_test/X_test.csv")
    print("   - data/train_test/y_train.csv")
    print("   - data/train_test/y_test.csv")
    print("   - models/preprocessing_pipeline.pkl")

    return X_train_processed, X_test_processed, y_train, y_test, feature_names


# ==========================================
# Run directly
# ==========================================

if __name__ == "__main__":
    preprocess_data("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")
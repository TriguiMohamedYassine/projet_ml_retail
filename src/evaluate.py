# ==========================================
# evaluate.py
# Model Evaluation & Visualization
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)

plt.style.use('seaborn-v0_8-whitegrid')


# ==========================================
# 1️⃣ Load Model and Data
# ==========================================

def load_model_and_data():
    """Load trained model and test data"""
    
    model = joblib.load("models/best_model.pkl")
    X_test = pd.read_csv("data/train_test/X_test.csv")
    y_test = pd.read_csv("data/train_test/y_test.csv").values.ravel()
    
    print(f"✅ Model loaded: {type(model).__name__}")
    print(f"✅ Test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    return model, X_test, y_test


# ==========================================
# 2️⃣ Calculate Metrics
# ==========================================

def calculate_metrics(model, X_test, y_test):
    """Calculate all evaluation metrics"""
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "Average Precision": average_precision_score(y_test, y_proba)
    }
    
    return metrics, y_pred, y_proba


# ==========================================
# 3️⃣ Visualizations
# ==========================================

def plot_confusion_matrix(y_test, y_pred, save_path="reports/confusion_matrix.png"):
    """Plot confusion matrix"""
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"📊 Saved: {save_path}")


def plot_roc_curve(y_test, y_proba, save_path="reports/roc_curve.png"):
    """Plot ROC curve"""
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#3498db', lw=2, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.2, color='#3498db')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"📊 Saved: {save_path}")


def plot_precision_recall_curve(y_test, y_proba, save_path="reports/pr_curve.png"):
    """Plot Precision-Recall curve"""
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='#2ecc71', lw=2, label=f'PR Curve (AP = {ap:.3f})')
    plt.fill_between(recall, precision, alpha=0.2, color='#2ecc71')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"📊 Saved: {save_path}")


def plot_feature_importance(model, X_test, top_n=20, save_path="reports/feature_importance.png"):
    """Plot feature importance"""
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X_test.columns
        
        # Get top N features
        indices = np.argsort(importances)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], color='#9b59b6')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"📊 Saved: {save_path}")
        
        # Return top features as DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    else:
        print("⚠️ Model doesn't have feature_importances_ attribute")
        return None


def plot_probability_distribution(y_test, y_proba, save_path="reports/prob_distribution.png"):
    """Plot probability distribution by actual class"""
    
    plt.figure(figsize=(10, 6))
    
    for label, color, name in [(0, '#2ecc71', 'No Churn'), (1, '#e74c3c', 'Churn')]:
        probs = y_proba[y_test == label]
        plt.hist(probs, bins=30, alpha=0.6, label=name, color=color, density=True)
    
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    plt.xlabel('Predicted Probability of Churn')
    plt.ylabel('Density')
    plt.title('Probability Distribution by Actual Class', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"📊 Saved: {save_path}")


# ==========================================
# 4️⃣ Generate Report
# ==========================================

def generate_report(metrics, importance_df=None, save_path="reports/evaluation_report.txt"):
    """Generate text evaluation report"""
    
    report = []
    report.append("="*60)
    report.append("MODEL EVALUATION REPORT")
    report.append("="*60)
    report.append("")
    report.append("PERFORMANCE METRICS")
    report.append("-"*40)
    
    for metric, value in metrics.items():
        report.append(f"{metric:.<25} {value:.4f}")
    
    if importance_df is not None:
        report.append("")
        report.append("TOP 10 IMPORTANT FEATURES")
        report.append("-"*40)
        for idx, row in importance_df.head(10).iterrows():
            report.append(f"{row['Feature']:.<35} {row['Importance']:.4f}")
    
    report.append("")
    report.append("="*60)
    
    report_text = "\n".join(report)
    
    with open(save_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n📄 Report saved: {save_path}")


# ==========================================
# 5️⃣ Main Execution
# ==========================================

def main():
    """Main evaluation pipeline"""
    
    # Create reports directory
    Path("reports").mkdir(exist_ok=True)
    
    # Load model and data
    model, X_test, y_test = load_model_and_data()
    
    # Calculate metrics
    metrics, y_pred, y_proba = calculate_metrics(model, X_test, y_test)
    
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION RESULTS")
    print("="*60)
    
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    # Generate visualizations
    print("\n📈 Generating visualizations...")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba)
    plot_precision_recall_curve(y_test, y_proba)
    importance_df = plot_feature_importance(model, X_test)
    plot_probability_distribution(y_test, y_proba)
    
    # Generate report
    generate_report(metrics, importance_df)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv("reports/metrics.csv", index=False)
    print("📊 Metrics saved: reports/metrics.csv")
    
    if importance_df is not None:
        importance_df.to_csv("reports/feature_importance.csv", index=False)
        print("📊 Feature importance saved: reports/feature_importance.csv")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()

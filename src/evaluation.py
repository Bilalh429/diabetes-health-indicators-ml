# src/evaluation.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, ConfusionMatrixDisplay,
    mean_absolute_error, mean_squared_error, r2_score
)

def plot_roc_curve(model, X_test, y_test, title="ROC Curve"):
    """Plot ROC curve for binary classification."""
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    return roc_auc

def plot_confusion_matrix(model, X_test, y_test, labels=None, title="Confusion Matrix"):
    """Plot confusion matrix."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return cm

def print_classification_metrics(model, X_test, y_test, task_name=""):
    """Print full classification report."""
    y_pred = model.predict(X_test)
    print(f"\n=== {task_name} Classification Report ===")
    print(classification_report(y_test, y_pred))

def print_regression_metrics(model, X_test, y_test, task_name=""):
    """Print regression metrics."""
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    
    print(f"\n=== {task_name} Regression Metrics ===")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

def plot_feature_importance(model, feature_names, title="Feature Importances", top_n=15):
    """Plot feature importances for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = abs(model.coef_[0]) if model.coef_.ndim > 1 else abs(model.coef_)
    else:
        print("Model doesn't support feature importance.")
        return
    
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(10, 6))
    plt.bar(range(top_n), importances[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def compare_models(results_dict, metric='Accuracy'):
    """Bar chart comparing model performance."""
    names  = list(results_dict.keys())
    values = [results_dict[n][metric] for n in names]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values, color=['steelblue', 'coral', 'seagreen'])
    plt.ylabel(metric)
    plt.title(f'Model Comparison — {metric}')
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.show()
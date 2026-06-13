# src/model_evaluation.py
# ─────────────────────────────────────────────────────────
# Reusable functions for evaluating trained models.
# AUC-PR is the primary metric because overall accuracy
# is misleading on imbalanced fraud datasets.
# ─────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.config import CV_FOLDS, RANDOM_STATE, PRIMARY_METRIC


def evaluate_model(model, X_test, y_test, model_name, dataset_name):
    """
    Evaluate a trained model and return all metrics.

    Primary metrics: AUC-PR and F1-Score
    Reason: accuracy is misleading on imbalanced data.
    A model predicting all transactions as legitimate achieves
    90.6% accuracy on Fraud_Data but catches ZERO fraud.

    Parameters
    ----------
    model        : trained sklearn/xgb/lgbm model
    X_test       : test features
    y_test       : true test labels
    model_name   : str  name for display
    dataset_name : str  dataset name for display

    Returns
    -------
    dict  containing all metrics and predictions
    """
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    f1       = f1_score(y_test, y_pred)
    roc_auc  = roc_auc_score(y_test, y_pred_prob)
    avg_prec = average_precision_score(y_test, y_pred_prob)
    cm       = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {model_name} — {dataset_name}")
    print(f"{'='*50}")
    print(f"  F1-Score : {f1:.4f}   ← primary metric")
    print(f"  AUC-PR   : {avg_prec:.4f}   ← primary metric")
    print(f"  ROC-AUC  : {roc_auc:.4f}   ← secondary metric")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legit','Fraud'])}")

    return {
        'model_name':       model_name,
        'dataset':          dataset_name,
        'f1':               round(f1, 4),
        'roc_auc':          round(roc_auc, 4),
        'auc_pr':           round(avg_prec, 4),
        'confusion_matrix': cm,
        'y_pred_prob':      y_pred_prob,
        'y_pred':           y_pred
    }


def evaluate_all_models(models_dict, X_test, y_test, dataset_name):
    """
    Evaluate all models and return sorted comparison DataFrame.

    Parameters
    ----------
    models_dict  : dict  {model_name: trained_model}
    X_test       : test features
    y_test       : true test labels
    dataset_name : str

    Returns
    -------
    results_list  : list of result dicts
    comparison_df : pd.DataFrame sorted by AUC-PR
    best_model    : the best performing trained model
    """
    results_list = []

    for name, model in models_dict.items():
        result = evaluate_model(
            model, X_test, y_test, name, dataset_name
        )
        results_list.append(result)

    comparison_df = pd.DataFrame([{
        'Model':   r['model_name'],
        'F1':      r['f1'],
        'ROC-AUC': r['roc_auc'],
        'AUC-PR':  r['auc_pr']
    } for r in results_list]).sort_values('AUC-PR', ascending=False)

    # Best model by primary metric (AUC-PR)
    best_result = max(results_list, key=lambda x: x[PRIMARY_METRIC])
    best_model  = models_dict[best_result['model_name']]

    print(f"\n🏆 Best model ({dataset_name}): "
          f"{best_result['model_name']} "
          f"| AUC-PR: {best_result['auc_pr']}")

    return results_list, comparison_df, best_model


def plot_confusion_matrices(results_list, save_path=None):
    """
    Plot confusion matrices for all models side by side.
    """
    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))

    for i, result in enumerate(results_list):
        sns.heatmap(
            result['confusion_matrix'],
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legit', 'Fraud'],
            yticklabels=['Legit', 'Fraud'],
            ax=axes[i]
        )
        axes[i].set_title(
            f"{result['model_name']}\n{result['dataset']}"
        )
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_pr_curves(results_list, y_test, dataset_name, save_path=None):
    """
    Plot Precision-Recall curves for all models.
    AUC-PR is shown in the legend for easy comparison.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for result in results_list:
        precision, recall, _ = precision_recall_curve(
            y_test, result['y_pred_prob']
        )
        ax.plot(
            recall, precision,
            label=f"{result['model_name']} (AUC-PR={result['auc_pr']})"
        )

    ax.set_title(f'Precision-Recall Curve — {dataset_name}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(fontsize=9)
    ax.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def run_cross_validation(models_dict, X_train, y_train):
    """
    Run Stratified K-Fold cross validation on all models.

    Stratified ensures each fold preserves the fraud ratio.
    Standard K-Fold on imbalanced data can create folds
    with very few or no fraud cases — making results unstable.

    Parameters
    ----------
    models_dict : dict  {model_name: untrained model instance}
    X_train     : training features
    y_train     : training labels

    Returns
    -------
    dict  {model_name: cv_scores_array}
    """
    skf = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    cv_results = {}

    print(f"\nRunning {CV_FOLDS}-Fold Cross Validation...")

    for name, model in models_dict.items():
        scores = cross_val_score(
            model, X_train, y_train,
            cv=skf, scoring='f1', n_jobs=-1
        )
        cv_results[name] = scores
        print(f"  {name:25s} F1: "
              f"{scores.mean():.4f} (+/- {scores.std():.4f})")

    return cv_results
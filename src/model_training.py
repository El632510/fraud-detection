# src/model_training.py
# ─────────────────────────────────────────────────────────
# Reusable functions for training all models.
# Hyperparameter choices are documented in src/config.py
# ─────────────────────────────────────────────────────────

import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.config import (
    LR_PARAMS, RF_PARAMS, XGB_PARAMS,
    LGBM_PARAMS, MODELS_DIR
)


def get_scale_pos_weight(y_train):
    """
    Calculate scale_pos_weight for XGBoost.
    This is XGBoost's native way to handle class imbalance.
    Formula: number of negatives / number of positives

    Parameters
    ----------
    y_train : array-like  training labels

    Returns
    -------
    float  scale_pos_weight value
    """
    neg = (np.array(y_train) == 0).sum()
    pos = (np.array(y_train) == 1).sum()
    return round(neg / pos, 2)


def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression as baseline model.
    Simple, fast, and interpretable — sets performance floor.
    """
    model = LogisticRegression(**LR_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """
    Train Random Forest ensemble model.
    Handles non-linearity and is robust to outliers.
    max_depth=10 prevents overfitting.
    """
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """
    Train XGBoost gradient boosting model.
    scale_pos_weight set dynamically to handle imbalance.
    Best performer on tabular fraud detection tasks.
    """
    spw = get_scale_pos_weight(y_train)
    params = {**XGB_PARAMS, 'scale_pos_weight': spw}
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train):
    """
    Train LightGBM model.
    Faster than XGBoost on large datasets due to
    leaf-wise tree growth strategy.
    """
    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_all_models(X_train, y_train, dataset_label=''):
    """
    Train all 4 models and return as dictionary.

    Parameters
    ----------
    X_train       : training features
    y_train       : training labels
    dataset_label : str  used for print messages

    Returns
    -------
    dict  {model_name: trained_model}
    """
    models = {}

    print(f"\nTraining models for {dataset_label}...")

    print("  [1/4] Logistic Regression...")
    models['Logistic Regression'] = train_logistic_regression(
        X_train, y_train
    )

    print("  [2/4] Random Forest...")
    models['Random Forest'] = train_random_forest(
        X_train, y_train
    )

    print("  [3/4] XGBoost...")
    models['XGBoost'] = train_xgboost(X_train, y_train)

    print("  [4/4] LightGBM...")
    models['LightGBM'] = train_lightgbm(X_train, y_train)

    print(f"  ✅ All 4 models trained for {dataset_label}!")
    return models


def save_models(models_dict, suffix=''):
    """
    Save all trained models to models/ directory.

    Parameters
    ----------
    models_dict : dict   {model_name: trained_model}
    suffix      : str    e.g. 'fraud' or 'cc'
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    for name, model in models_dict.items():
        filename = name.lower().replace(' ', '_')
        path = os.path.join(MODELS_DIR, f'{filename}_{suffix}.pkl')
        joblib.dump(model, path)
        print(f"  Saved: {path}")


def save_best_model(model, suffix='fraud'):
    """
    Save the best model separately for easy loading in Task 3.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f'best_model_{suffix}.pkl')
    joblib.dump(model, path)
    print(f"✅ Best model saved: {path}")
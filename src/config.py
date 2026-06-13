# src/config.py
# ─────────────────────────────────────────────────────────
# Central configuration file for the fraud detection project
# All hyperparameters and settings live here so they are
# easy to find, change, and document in one place.
# ─────────────────────────────────────────────────────────

import os

# ── File Paths ────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW       = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR     = os.path.join(BASE_DIR, 'models')

FRAUD_RAW_PATH      = os.path.join(DATA_RAW, 'Fraud_Data.csv')
IP_RAW_PATH         = os.path.join(DATA_RAW, 'IpAddress_to_Country.csv')
CREDITCARD_RAW_PATH = os.path.join(DATA_RAW, 'creditcard.csv')

# ── Random Seed ───────────────────────────────────────────
# Fixed seed ensures results are reproducible across runs
RANDOM_STATE = 42

# ── Train/Test Split ──────────────────────────────────────
# 80/20 split is standard for this dataset size
# Stratified to preserve fraud ratio in both sets
TEST_SIZE = 0.2

# ── SMOTE ─────────────────────────────────────────────────
# Applied on training set ONLY to avoid data leakage
# Oversamples minority (fraud) class to 1:1 ratio
SMOTE_RANDOM_STATE = 42

# ── Logistic Regression Hyperparameters ───────────────────
# Used as BASELINE model — simple and interpretable
# max_iter=1000: default 100 is too low for this dataset size
# class_weight='balanced': adjusts for class imbalance
#   without needing SMOTE (used as extra safeguard)
# solver='lbfgs': efficient for medium-sized datasets
LR_PARAMS = {
    'max_iter':     1000,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced',
    'solver':       'lbfgs'
}

# ── Random Forest Hyperparameters ─────────────────────────
# n_estimators=100: enough trees for stable predictions
#   without excessive training time
# max_depth=10: prevents overfitting on training data
#   (unlimited depth memorizes noise)
# class_weight='balanced': handles class imbalance
# n_jobs=-1: uses all CPU cores for faster training
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth':    10,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced',
    'n_jobs':       -1
}

# ── XGBoost Hyperparameters ───────────────────────────────
# n_estimators=100: number of boosting rounds
# max_depth=6: shallower than RF to reduce overfitting
#   in boosted trees (standard starting point)
# learning_rate=0.1: step size shrinkage — lower = more
#   robust but slower. 0.1 is the industry standard default
# scale_pos_weight: set at runtime = neg/pos ratio
#   this is XGBoost's native way to handle imbalance
# eval_metric='logloss': appropriate for binary classification
XGB_PARAMS = {
    'n_estimators':  100,
    'max_depth':     6,
    'learning_rate': 0.1,
    'random_state':  RANDOM_STATE,
    'eval_metric':   'logloss',
    'verbosity':     0
}

# ── LightGBM Hyperparameters ──────────────────────────────
# Same structure as XGBoost for fair comparison
# LightGBM is faster on large datasets (leaf-wise growth)
# class_weight='balanced': handles imbalance natively
# verbosity=-1: suppresses training output noise
LGBM_PARAMS = {
    'n_estimators':  100,
    'max_depth':     6,
    'learning_rate': 0.1,
    'random_state':  RANDOM_STATE,
    'class_weight':  'balanced',
    'verbosity':     -1
}

# ── Cross Validation ──────────────────────────────────────
# 5 folds is standard — enough folds for stable estimate
# without excessive computation time
# Stratified ensures each fold has same fraud % as full set
CV_FOLDS = 5

# ── Decision Threshold ────────────────────────────────────
# Default is 0.5 but lowering to 0.3 increases recall
# on fraud cases — catching more fraud at cost of slightly
# more false positives. Justified because financial cost
# of missed fraud > cost of customer friction.
DECISION_THRESHOLD = 0.3

# ── Primary Evaluation Metric ─────────────────────────────
# AUC-PR chosen over accuracy because both datasets are
# highly imbalanced. Accuracy is misleading — a model
# predicting all transactions as legit gets 90.6% accuracy
# but catches zero fraud. AUC-PR focuses on the minority class.
PRIMARY_METRIC = 'auc_pr'
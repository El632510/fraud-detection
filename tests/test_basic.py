# tests/test_basic.py
# ─────────────────────────────────────────────────────────
# Unit tests for src/ modules
# Run with: python -m pytest tests/ -v
# ─────────────────────────────────────────────────────────

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Test imports ──────────────────────────────────────────
def test_imports():
    """All core libraries must be importable"""
    import pandas
    import numpy
    import sklearn
    import xgboost
    import lightgbm
    import shap
    assert True


def test_src_imports():
    """All src modules must be importable"""
    from src.data_preprocessing import (
        clean_fraud_data,
        engineer_features,
        encode_and_scale_fraud
    )
    from src.model_training import (
        train_logistic_regression,
        get_scale_pos_weight
    )
    from src.model_evaluation import evaluate_model
    assert True


# ── Test data preprocessing functions ────────────────────
def test_clean_fraud_data():
    """clean_fraud_data should remove duplicates and fix types"""
    from src.data_preprocessing import clean_fraud_data

    raw = pd.DataFrame({
        'signup_time':   ['2023-01-01 10:00:00', '2023-01-01 10:00:00'],
        'purchase_time': ['2023-01-01 11:00:00', '2023-01-01 11:00:00'],
        'user_id':       [1, 1],
        'ip_address':    [1234567.0, 1234567.0],
        'device_id':     ['abc', 'abc'],
        'source':        ['SEO', 'SEO'],
        'browser':       ['Chrome', 'Chrome'],
        'sex':           ['M', 'M'],
        'age':           [25, 25],
        'purchase_value':[100, 100],
        'class':         [0, 0]
    })

    cleaned = clean_fraud_data(raw)
    # Duplicate should be removed
    assert len(cleaned) == 1
    # Timestamps should be datetime
    assert pd.api.types.is_datetime64_any_dtype(
        cleaned['signup_time']
    )


def test_engineer_features():
    """engineer_features should add 4 new columns"""
    from src.data_preprocessing import (
        clean_fraud_data, engineer_features
    )

    raw = pd.DataFrame({
        'signup_time':   ['2023-01-01 10:00:00'],
        'purchase_time': ['2023-01-01 11:30:00'],
        'user_id':       [1],
        'ip_address':    [1234567.0],
        'device_id':     ['abc'],
        'source':        ['SEO'],
        'browser':       ['Chrome'],
        'sex':           ['M'],
        'age':           [25],
        'purchase_value':[100],
        'class':         [0]
    })

    cleaned     = clean_fraud_data(raw)
    engineered  = engineer_features(cleaned)

    assert 'time_since_signup'      in engineered.columns
    assert 'hour_of_day'            in engineered.columns
    assert 'day_of_week'            in engineered.columns
    assert 'user_transaction_count' in engineered.columns

    # time_since_signup should be 1.5 hours
    assert abs(engineered['time_since_signup'].iloc[0] - 1.5) < 0.01


def test_get_scale_pos_weight():
    """scale_pos_weight should equal neg/pos ratio"""
    from src.model_training import get_scale_pos_weight

    y = np.array([0, 0, 0, 0, 1, 1])
    # 4 negatives, 2 positives → ratio = 2.0
    spw = get_scale_pos_weight(y)
    assert spw == 2.0


def test_config_values():
    """Config values should be valid"""
    from src.config import (
        TEST_SIZE, CV_FOLDS,
        RANDOM_STATE, DECISION_THRESHOLD
    )

    assert 0 < TEST_SIZE < 1
    assert CV_FOLDS >= 3
    assert RANDOM_STATE == 42
    assert 0 < DECISION_THRESHOLD < 1
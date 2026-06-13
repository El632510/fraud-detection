# src/data_preprocessing.py
# ─────────────────────────────────────────────────────────
# Reusable functions for data cleaning, feature engineering,
# geolocation merge, encoding, scaling, and SMOTE.
# These were extracted from eda-fraud-data.ipynb so they
# can be reused, tested, and deployed in production.
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.config import (
    FRAUD_RAW_PATH, IP_RAW_PATH, CREDITCARD_RAW_PATH,
    RANDOM_STATE, TEST_SIZE, SMOTE_RANDOM_STATE
)


def load_data():
    """
    Load all three raw datasets from data/raw/.

    Returns
    -------
    fraud_df : pd.DataFrame  — e-commerce transactions
    ip_df    : pd.DataFrame  — IP to country mapping
    cc_df    : pd.DataFrame  — credit card transactions
    """
    fraud_df = pd.read_csv(FRAUD_RAW_PATH)
    ip_df    = pd.read_csv(IP_RAW_PATH)
    cc_df    = pd.read_csv(CREDITCARD_RAW_PATH)
    return fraud_df, ip_df, cc_df


def clean_fraud_data(fraud_df):
    """
    Clean Fraud_Data.csv:
    - Convert timestamps to datetime
    - Remove duplicates
    - Drop nulls

    Parameters
    ----------
    fraud_df : pd.DataFrame  raw fraud data

    Returns
    -------
    pd.DataFrame  cleaned fraud data
    """
    df = fraud_df.copy()

    # Convert to datetime for time-based feature engineering
    df['signup_time']   = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])

    df = df.drop_duplicates()
    df = df.dropna()

    return df


def clean_creditcard_data(cc_df):
    """
    Clean creditcard.csv:
    - Remove duplicates
    - Drop nulls
    V1-V28 are already PCA-scaled so no type conversion needed.

    Parameters
    ----------
    cc_df : pd.DataFrame  raw creditcard data

    Returns
    -------
    pd.DataFrame  cleaned creditcard data
    """
    df = cc_df.copy()
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def merge_ip_to_country(fraud_df, ip_df):
    """
    Add country column to fraud_df using IP address ranges.

    IP addresses in Fraud_Data are stored as float integers.
    IpAddress_to_Country.csv has lower/upper bound ranges.
    We use merge_asof (range-based merge) because each IP
    must be matched to a range, not an exact value.

    Parameters
    ----------
    fraud_df : pd.DataFrame  cleaned fraud data
    ip_df    : pd.DataFrame  IP range to country mapping

    Returns
    -------
    pd.DataFrame  fraud data with country column added
    """
    df = fraud_df.copy()

    # IP addresses already stored as float — use directly
    df['ip_int'] = pd.to_numeric(df['ip_address'], errors='coerce')
    df = df.dropna(subset=['ip_int'])

    # Sort both for merge_asof requirement
    ip_sorted    = ip_df.sort_values('lower_bound_ip_address').reset_index(drop=True)
    fraud_sorted = df.sort_values('ip_int').reset_index(drop=True)

    # Range-based merge
    merged = pd.merge_asof(
        fraud_sorted,
        ip_sorted[['lower_bound_ip_address',
                   'upper_bound_ip_address', 'country']],
        left_on='ip_int',
        right_on='lower_bound_ip_address',
        direction='backward'
    )

    # Remove IPs outside upper bound
    merged = merged[
        merged['ip_int'] <= merged['upper_bound_ip_address']
    ]

    # Fill unmatched IPs
    merged['country'] = merged['country'].fillna('Unknown')

    return merged


def engineer_features(fraud_df):
    """
    Create new features from existing columns in Fraud_Data.

    Features created:
    - time_since_signup: hours between signup and purchase
      Fraud hypothesis: fraudsters buy immediately after signup
    - hour_of_day: hour of purchase (0-23)
      Fraud hypothesis: fraud clusters at unusual hours
    - day_of_week: day of purchase (0=Mon, 6=Sun)
      Fraud hypothesis: weekend fraud slightly higher
    - user_transaction_count: number of transactions per user
      Fraud hypothesis: high velocity = card testing

    Parameters
    ----------
    fraud_df : pd.DataFrame  cleaned fraud data with timestamps

    Returns
    -------
    pd.DataFrame  fraud data with new features added
    """
    df = fraud_df.copy()

    # Time since signup in hours
    df['time_since_signup'] = (
        df['purchase_time'] - df['signup_time']
    ).dt.total_seconds() / 3600

    # Temporal features from purchase timestamp
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek

    # Transaction velocity per user
    tx_count = (
        df.groupby('user_id')['purchase_time']
          .count()
          .reset_index()
    )
    tx_count.columns = ['user_id', 'user_transaction_count']
    df = df.merge(tx_count, on='user_id', how='left')

    return df


def encode_and_scale_fraud(fraud_df):
    """
    Prepare Fraud_Data for modeling:
    - Drop identifier and datetime columns
    - One-hot encode categorical features
    - StandardScale numerical features

    Parameters
    ----------
    fraud_df : pd.DataFrame  fraud data with engineered features

    Returns
    -------
    fraud_model : pd.DataFrame  model-ready fraud data
    feature_cols : list         list of feature column names
    scaler : StandardScaler     fitted scaler for reuse
    """
    df = fraud_df.copy()

    # Drop columns not useful for modeling
    drop_cols = [
        'user_id', 'device_id', 'ip_address',
        'signup_time', 'purchase_time',
        'ip_int', 'lower_bound_ip_address',
        'upper_bound_ip_address', 'country'
    ]
    df = df.drop(columns=drop_cols, errors='ignore')

    # One-hot encode — drop_first avoids multicollinearity
    cat_cols = ['source', 'browser', 'sex']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Scale numerical features to zero mean, unit variance
    # Required so large-value features don't dominate
    num_cols = [
        'purchase_value', 'age', 'time_since_signup',
        'hour_of_day', 'day_of_week', 'user_transaction_count'
    ]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    feature_cols = [c for c in df.columns if c != 'class']
    return df, feature_cols, scaler


def scale_creditcard(cc_df):
    """
    Scale Amount and Time in creditcard dataset.
    V1-V28 are already PCA-normalized so only these
    two columns need scaling.

    Parameters
    ----------
    cc_df : pd.DataFrame  cleaned creditcard data

    Returns
    -------
    pd.DataFrame  creditcard data with scaled Amount and Time
    StandardScaler fitted scaler
    """
    df = cc_df.copy()
    scaler = StandardScaler()
    df[['Amount', 'Time']] = scaler.fit_transform(
        df[['Amount', 'Time']]
    )
    return df, scaler


def split_and_resample(X, y, label='dataset'):
    """
    Split into train/test and apply SMOTE on training set only.

    SMOTE is applied ONLY on training data to prevent leakage.
    Applying SMOTE on test data would give inflated metrics
    that do not reflect real-world performance.

    Parameters
    ----------
    X     : pd.DataFrame or np.ndarray  features
    y     : pd.Series or np.ndarray     target
    label : str                         dataset name for logging

    Returns
    -------
    X_train_sm, y_train_sm : SMOTE-resampled training data
    X_test, y_test         : original imbalanced test data
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"{label} — before SMOTE fraud %: "
          f"{round(y_train.mean()*100, 2)}%")

    smote = SMOTE(random_state=SMOTE_RANDOM_STATE)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    print(f"{label} — after SMOTE fraud %:  "
          f"{round(y_train_sm.mean()*100, 2)}%")

    return X_train_sm, y_train_sm, X_test, y_test
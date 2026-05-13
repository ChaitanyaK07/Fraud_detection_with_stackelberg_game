"""
=============================================================================
FILE 1: DATA PROCESSING PIPELINE
=============================================================================
Deep Learning-Based Fraud Detection in Banking and UPI Transactions
Advisor: Aneesh Chivukula

LEAKAGE AUDIT RESULTS (verified column-by-column):
The dataset is synthetic. The following features encode the label directly
and are EXCLUDED:
  - handle_verification_status     : legit=verified (100%), fraud=unverified (100%)
  - unusual_transaction_amount_flag : legit always=0
  - otp_request_device_consistency  : legit always=1
  - session_source                  : 'link' category = 100% fraud
  - pin_entry_method                : 'pasted' category = 100% fraud
  - authorization_method, transaction_type: same pattern
  - unusual_device/ip/location_flag : legit always=0
  - authentication_attempts/count   : legit always fixed value
  - dns_lookup_age, otp_request_frequency, ... (21 binary flags total)
  - receiver_account_age            : fraud always=0
  - All categorical features        : one category always maps to one class
  - Engineered composite_risk_score : aggregates leaking flags

RETAINED (14 numeric features where both classes genuinely vary):
  amount, session_duration, receiver_transaction_history,
  transaction_amount_vs_sender_history, transaction_time_of_day,
  input_timing_consistency, app_switching_frequency, keyboard_input_speed,
  input_pause_patterns, screen_active_time, background_data_usage,
  pin_entry_speed, request_amount_roundness, geographic_disparity

Plus 4 engineered features derived solely from the above.
Max single-feature AUC after cleaning: ~0.95 (receiver_transaction_history
  has genuine distributional separation but both classes overlap fully).
=============================================================================
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
import os, pickle

DATA_PATH   = "fraud_dataset.csv"
OUTPUT_DIR  = "processed_data"
RANDOM_SEED = 42
TEST_SIZE   = 0.15
VAL_SIZE    = 0.15
TARGET_COL  = 'is_fraud'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Verified clean features: both classes exhibit genuine distributional overlap
CLEAN_FEATURES = [
    'amount',
    'session_duration',
    'receiver_transaction_history',
    'transaction_amount_vs_sender_history',
    'transaction_time_of_day',
    'input_timing_consistency',
    'app_switching_frequency',
    'keyboard_input_speed',
    'input_pause_patterns',
    'screen_active_time',
    'background_data_usage',
    'pin_entry_speed',
    'request_amount_roundness',
    'geographic_disparity',
]


def load_data(path: str) -> pd.DataFrame:
    print(f"[1/5] Loading: {path}")
    df = pd.read_csv(path)
    print(f"      Raw shape: {df.shape} | Fraud rate: {df[TARGET_COL].mean()*100:.2f}%")
    return df


def select_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/5] Selecting verified non-leaking features ...")
    available = [c for c in CLEAN_FEATURES if c in df.columns]
    df = df[available + [TARGET_COL]].copy()
    for col in available:
        df[col] = df[col].fillna(df[col].median())
    print(f"      Features: {len(available)} | NaN: {df.isnull().sum().sum()}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/5] Engineering features ...")
    df = df.copy()

    # Log-transform heavy-tailed amount
    df['log_amount'] = np.log1p(df['amount'])

    # Behavioral typing anomaly: high pauses relative to speed = scripted
    df['typing_anomaly'] = (
        df['input_pause_patterns'] / (df['keyboard_input_speed'] + 1e-6)
    )

    # Amount z-score relative to sender's own history
    df['amount_vs_history_ratio'] = (
        df['amount'] / (df['transaction_amount_vs_sender_history'] + 1e-6)
    )

    # Session engagement: long screen time relative to session = legitimate
    df['engagement_ratio'] = (
        df['screen_active_time'] / (df['session_duration'] + 1.0)
    )

    print(f"      Total features: {df.shape[1] - 1}")
    return df


def split_scale_save(df: pd.DataFrame) -> dict:
    print("[4/5] Splitting, scaling, saving ...")
    feature_names = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_names].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int64)

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    val_frac = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, random_state=RANDOM_SEED, stratify=y_tv
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train.astype(np.float32))
    np.save(f"{OUTPUT_DIR}/X_val.npy",   X_val.astype(np.float32))
    np.save(f"{OUTPUT_DIR}/X_test.npy",  X_test.astype(np.float32))
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_val.npy",   y_val)
    np.save(f"{OUTPUT_DIR}/y_test.npy",  y_test)
    with open(f"{OUTPUT_DIR}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    with open(f"{OUTPUT_DIR}/feature_names.pkl", 'wb') as f:
        pickle.dump(feature_names, f)

    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    np.save(f"{OUTPUT_DIR}/class_weights.npy", cw)

    print(f"      Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")
    print(f"      Dimensionality: {X_train.shape[1]}")
    print(f"      Class weights: 0={cw[0]:.3f}  1={cw[1]:.3f}")

    return dict(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        feature_names=feature_names, n_features=X_train.shape[1]
    )


def verify_no_leakage(data: dict):
    print("\n[VERIFY] Leakage checks on training split ...")
    X, y = data['X_train'], data['y_train']
    fn   = data['feature_names']

    fixed = [fn[i] for i in range(X.shape[1])
             if len(np.unique(X[y == 0, i])) == 1
             or len(np.unique(X[y == 1, i])) == 1]
    if fixed:
        print(f"  [WARN] Fixed-value features: {fixed}")
    else:
        print("  [OK] All features vary within both classes.")

    aucs = []
    for i in range(X.shape[1]):
        try:
            a = roc_auc_score(y, X[:, i])
            aucs.append(max(a, 1 - a))
        except Exception:
            pass
    print(f"  Max single-feature AUC  : {max(aucs):.4f}")
    print(f"  Mean single-feature AUC : {np.mean(aucs):.4f}")
    if max(aucs) >= 0.999:
        print("  [WARN] Perfect predictor detected.")
    else:
        print("  [OK] No perfect single-feature predictor.")


def run_pipeline(path: str = DATA_PATH) -> dict:
    print("=" * 70)
    print("DATA PROCESSING PIPELINE -- Fraud Detection (Banking + UPI)")
    print("=" * 70)
    df   = load_data(path)
    df   = select_and_clean(df)
    df   = engineer_features(df)
    data = split_scale_save(df)
    verify_no_leakage(data)
    print("=" * 70)
    print("[DONE] Data processing complete.")
    print("=" * 70)
    return data


if __name__ == "__main__":
    run_pipeline()

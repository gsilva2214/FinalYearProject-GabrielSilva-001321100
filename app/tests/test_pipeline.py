"""
Check file existence and column contracts.
"""
import os
import json
import pandas as pd
import pytest
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core import PATHS

ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANOMALY_TABS  = os.path.join(ROOT, "outputs", "anomaly", "tables")
SNORT_TABS    = os.path.join(ROOT, "outputs", "snort",   "tables")

# file checks

def test_anomaly_metrics_json_exists():
    path = os.path.join(ANOMALY_TABS, "anomaly_metrics.json")
    assert os.path.exists(path), f"Missing: {path} — run anomaly_train_score.py"

def test_snort_metrics_json_exists():
    path = os.path.join(SNORT_TABS, "snort_metrics.json")
    assert os.path.exists(path), f"Missing: {path} — run snort_evaluate.py"

def test_anomaly_predictions_csv_exists():
    path = os.path.join(ANOMALY_TABS, "anomaly_predictions.csv")
    assert os.path.exists(path), f"Missing: {path} — run anomaly_train_score.py"

def test_snort_predictions_csv_exists():
    path = os.path.join(SNORT_TABS, "snort_predictions.csv")
    assert os.path.exists(path), f"Missing: {path} — run snort_evaluate.py"

def test_anomaly_metrics_schema():
    path = os.path.join(ANOMALY_TABS, "anomaly_metrics.json")
    if not os.path.exists(path):
        pytest.skip("anomaly_metrics.json not yet generated")
    with open(path) as f:
        m = json.load(f)
    required = ["precision", "recall", "f1_score", "false_positive_rate",
                 "true_positives", "false_positives", "true_negatives", "false_negatives"]
    for key in required:
        assert key in m, f"Missing key '{key}' in anomaly_metrics.json"
        assert 0.0 <= float(m[key]) <= max(float(m[key]), 1.0), f"Out of range: {key}={m[key]}"

def test_snort_metrics_schema():
    path = os.path.join(SNORT_TABS, "snort_metrics.json")
    if not os.path.exists(path):
        pytest.skip("snort_metrics.json not yet generated")
    with open(path) as f:
        m = json.load(f)
    required = ["precision", "recall", "f1_score", "false_positive_rate"]
    for key in required:
        assert key in m, f"Missing key '{key}' in snort_metrics.json"

def test_anomaly_predictions_columns():
    path = os.path.join(ANOMALY_TABS, "anomaly_predictions.csv")
    if not os.path.exists(path):
        pytest.skip("anomaly_predictions.csv not yet generated")
    df = pd.read_csv(path, nrows=5)
    for col in ["label_binary", "ml_prediction", "label_original"]:
        assert col in df.columns, f"Missing column '{col}' in anomaly_predictions.csv"

def test_snort_predictions_columns():
    path = os.path.join(SNORT_TABS, "snort_predictions.csv")
    if not os.path.exists(path):
        pytest.skip("snort_predictions.csv not yet generated")
    df = pd.read_csv(path, nrows=5)
    for col in ["label_binary", "snort_prediction"]:
        assert col in df.columns, f"Missing column '{col}' in snort_predictions.csv"

def test_prediction_binary_values():
    """Predictions must be 0 or 1 only."""
    for name, path, col in [
        ("anomaly", os.path.join(ANOMALY_TABS, "anomaly_predictions.csv"), "ml_prediction"),
        ("snort",   os.path.join(SNORT_TABS,   "snort_predictions.csv"),   "snort_prediction"),
    ]:
        if not os.path.exists(path):
            pytest.skip(f"{path} not yet generated")
        df = pd.read_csv(path, nrows=1000)
        unique_vals = set(df[col].unique())
        assert unique_vals.issubset({0, 1}), f"{name} {col} contains non-binary values: {unique_vals}"

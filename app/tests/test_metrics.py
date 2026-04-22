
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.metrics import (
    calculate_all_metrics,
    calculate_false_positive_rate,
    calculate_confusion_matrix,
)

def test_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    m = calculate_all_metrics(y, y)
    assert m["precision"] == 1.0
    assert m["recall"]    == 1.0
    assert m["f1"]        == 1.0
    assert m["fpr"]       == 0.0
    assert m["accuracy"]  == 1.0

def test_all_wrong():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    m = calculate_all_metrics(y_true, y_pred)
    assert m["precision"] == 0.0
    assert m["recall"]    == 0.0
    assert m["fpr"]       == 1.0

def test_all_predict_attack():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 1, 1])
    m = calculate_all_metrics(y_true, y_pred)
    assert m["recall"] == 1.0    # catch all attacks
    assert m["fpr"]    == 1.0    # flag all benign too

def test_fpr_no_benign():
    # No true negatives → FPR should not crash
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 0])
    fpr = calculate_false_positive_rate(y_true, y_pred)
    assert fpr == 0.0

def test_confusion_matrix_shape():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    cm = calculate_confusion_matrix(y_true, y_pred, labels=[0, 1])
    assert cm.shape == (2, 2)

def test_f1_partial():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    m = calculate_all_metrics(y_true, y_pred)
    assert 0 < m["f1"] < 1
    assert 0 < m["fpr"] < 1

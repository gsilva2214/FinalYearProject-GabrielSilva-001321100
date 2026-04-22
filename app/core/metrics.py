import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

def calculate_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    return pd.DataFrame(cm, index=labels, columns=labels)

def calculate_false_positive_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    if (fp + tn) == 0:
        return 0.0
    return fp / (fp + tn)

def calculate_all_metrics(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "fpr": calculate_false_positive_rate(y_true, y_pred),
    }

def calculate_per_attack_metrics(df, label_col, pred_col):
    results = []
    for attack_type in df[label_col].unique():
        subset = df[df[label_col] == attack_type]
        y_true = (subset[label_col] != "BENIGN").astype(int)
        y_pred = subset[pred_col].astype(int)
        if len(y_true) == 0:
            continue
        metrics = calculate_all_metrics(y_true.values, y_pred.values)
        metrics["attack_type"] = attack_type
        metrics["count"] = len(subset)
        results.append(metrics)
    return pd.DataFrame(results).set_index("attack_type")
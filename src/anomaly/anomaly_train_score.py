from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)
import joblib
import time
import json

ROOT = Path(__file__).resolve().parents[2]
TAB_DIR = ROOT / "outputs" / "anomaly" / "tables"
FIG_DIR = ROOT / "outputs" / "anomaly" / "figures"
MODEL_DIR = ROOT / "models"                          # where model is saved
TAB_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)       

RANDOM_SEED = 42
CONTAMINATION = 0.1  # cic ids is roughly 10-20% attacks


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = {"label_binary", "label_original"}
    return [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]


def main() -> None:

    # loads cleaned features
    features_csv = TAB_DIR.parent / "anomaly_results.csv"
    if not features_csv.exists():
        raise FileNotFoundError(
            f"Missing: {features_csv}. "
            f"Run anomaly_features.py first."
        )

    print(f"Loading: {features_csv}")
    df = pd.read_csv(features_csv, low_memory=False)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df["label_binary"].values

    print(f"Samples: {len(df)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Attacks: {y.sum()}")
    print(f"Benign: {(y == 0).sum()}")

    # SCALE features

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # TRAIN Isolation Forest

    print("Training Isolation Forest...")
    start_train = time.time()

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    train_time = time.time() - start_train
    print(f"Training time: {train_time:.2f}s") # saves model + scaler + feature list
    joblib.dump(model,   MODEL_DIR / "isolation_forest.joblib")
    joblib.dump(scaler,  MODEL_DIR / "scaler.joblib") # saves the exact feature column names so the live detector
    feature_meta = {
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "contamination": CONTAMINATION,
        "n_estimators": 200,
        "random_seed": RANDOM_SEED,
    }
    with open(MODEL_DIR / "feature_meta.json", "w") as f:
        json.dump(feature_meta, f, indent=2)

    print(f"Model saved  → {MODEL_DIR / 'isolation_forest.joblib'}")
    print(f"Scaler saved → {MODEL_DIR / 'scaler.joblib'}")
    print(f"Features     → {MODEL_DIR / 'feature_meta.json'}")

    start_pred = time.time()

    raw_preds = model.predict(X_scaled)     # isolationForest: 1 = normal, -1 = anomaly
    predictions = np.where(raw_preds == -1, 1, 0)

    anomaly_scores = -model.decision_function(X_scaled)

    pred_time = time.time() - start_pred
    print(f"Prediction time: {pred_time:.2f}s")

    # evaluates against ground truth
    tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
    fpr = fp / (fp + tn)
    detection_rate = tp / (tp + fn)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)

    print("\n=== RESULTS ===")
    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives:  {tn}")
    print(f"False Negatives: {fn}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"F1 Score:        {f1:.4f}")
    print(f"FPR:             {fpr:.4f}")
    print(f"Detection Rate:  {detection_rate:.4f}")

    # saves metrics
    metrics = {
        "model": "IsolationForest",
        "n_estimators": 200,
        "contamination": CONTAMINATION,
        "total_samples": int(len(y)),
        "attack_samples": int(y.sum()),
        "benign_samples": int((y == 0).sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "detection_rate": round(detection_rate, 4),
        "train_time_seconds": round(train_time, 2),
        "prediction_time_seconds": round(pred_time, 2),
        "feature_count": len(feature_cols),
    }

    metrics_path = TAB_DIR / "anomaly_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    metrics_json = TAB_DIR / "anomaly_metrics.json"
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    attack_perf = []     # per attack type performance
    for attack_type in df["label_original"].unique():
        mask = df["label_original"] == attack_type
        if mask.sum() == 0:
            continue
        a_truth = y[mask]
        a_preds = predictions[mask]
        detected = int((a_preds == 1).sum())
        total = int(mask.sum())
        attack_perf.append({
            "attack_type": attack_type,
            "total": total,
            "detected_as_anomaly": detected,
            "detection_rate": round(detected / total * 100, 2),
        })

    attack_df = pd.DataFrame(attack_perf).sort_values(
        "detection_rate", ascending=False
    )
    attack_df.to_csv(TAB_DIR / "per_attack_detection.csv", index=False)
    print("\nPer attack type detection:")
    print(attack_df.to_string(index=False))

    results_df = df[["label_original", "label_binary"]].copy()     # save predictions for comparison stage
    results_df["ml_prediction"] = predictions
    results_df["anomaly_score"] = anomaly_scores
    results_df.to_csv(TAB_DIR / "anomaly_predictions.csv", index=False)

    plt.figure(figsize=(8, 6))     # confusion Matrix
    cm = confusion_matrix(y, predictions)
    labels = [[f"{val:,}" for val in row] for row in cm]
    labels = np.array(labels)
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=["BENIGN", "ATTACK"],
        yticklabels=["BENIGN", "ATTACK"],
        cbar=False,
    )
    plt.title("Anomaly Detection - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_confusion_matrix.png", dpi=220)
    plt.close()

    plt.figure(figsize=(12, 6))     # per Attack Type Detection Rate
    attack_plot = attack_df[attack_df["attack_type"] != "BENIGN"].copy()
    plt.barh(attack_plot["attack_type"], attack_plot["detection_rate"])
    plt.title("Detection Rate by Attack Type")
    plt.xlabel("Detection Rate (%)")
    plt.ylabel("Attack Type")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_per_attack_detection_rate.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5)) # anomaly score distribution
    plt.hist(anomaly_scores[y == 0], bins=100, alpha=0.5, label="BENIGN", density=True)
    plt.hist(anomaly_scores[y == 1], bins=100, alpha=0.5, label="ATTACK", density=True)
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_score_distribution.png", dpi=220)
    plt.close()

    fpr_curve, tpr_curve, _ = roc_curve(y, anomaly_scores)     # ROC Curve
    roc_auc = auc(fpr_curve, tpr_curve)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_curve, tpr_curve, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("Anomaly Detection - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_roc_curve.png", dpi=220)
    plt.close()

    print(f"\nAll outputs saved to {TAB_DIR} and {FIG_DIR}")
    print(f"Model files saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
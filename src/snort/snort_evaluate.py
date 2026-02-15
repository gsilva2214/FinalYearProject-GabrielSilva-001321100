from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
import json
import time

ROOT = Path(__file__).resolve().parents[2]
SNORT_DIR = ROOT / "outputs" / "snort"
TAB_DIR = SNORT_DIR / "tables"
FIG_DIR = SNORT_DIR / "figures"
TAB_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SNORT_ALERTS_FULL = SNORT_DIR / "snort_alerts.csv"
CICIDS_CSV = ROOT / "outputs" / "anomaly" / "anomaly_results.csv" # changed from cicids_dataset to allow the comparison to be fair and check the same amount of data.

def load_cicids(path: Path) -> pd.DataFrame: #loads and cleans CIC IDS csv file.
    print(f"Loading CIC-IDS dataset: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    if "label_original" in df.columns: # already cleaned by anomaly_features.py file
        df["label_binary"] = df["label_binary"].astype(int)
    elif "Label" in df.columns:
        df["label_original"] = df["Label"].str.strip()
        df["label_binary"] = df["label_original"].apply(
            lambda x: 0 if x == "BENIGN" else 1
        )
    else:
        raise KeyError(
            f"No label column found. "
            f"Columns: {list(df.columns)}"
        )

    for col in ["Source IP", "Destination IP"]: # cleans IP and ports
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    for col in ["Source Port", "Destination Port"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )
    print(f"  Rows: {len(df)}")
    print(f"  Attacks: {df['label_binary'].sum()}")
    print(f"  Benign: {(df['label_binary'] == 0).sum()}")
    return df

def load_snort_alerts(path: Path) -> pd.DataFrame: # load the fully parsed snort alerts with IPs.
    print(f"Loading Snort alerts: {path}")
    df = pd.read_csv(path)
    print(f"  Total alerts: {len(df)}")
    return df

def map_snort_to_flows( #flow is marked as flagged by Snort if a Snort alert matches its IP pair and destination port on the same day. forward direction only.
    flows: pd.DataFrame,
    alerts: pd.DataFrame,
) -> tuple:

    print("Mapping Snort alerts to flows...")
    start = time.time()

    alerts = alerts.copy()
    flows = flows.copy()
    # Filter out IPv6
    before = len(alerts)
    alerts = alerts.dropna(subset=["src_ip", "dst_ip"])

    ipv4_mask = (
        ~alerts["src_ip"].str.contains("::", na=True) &
        ~alerts["dst_ip"].str.contains("::", na=True) &
        ~alerts["src_ip"].str.contains(
            "[a-fA-F]", regex=True, na=True
        ) &
        ~alerts["dst_ip"].str.contains(
            "[a-fA-F]", regex=True, na=True
        )
    )
    alerts = alerts[ipv4_mask].copy()
    print(f"  IPv4 alerts: {len(alerts)}/{before}")

    alerts = alerts.dropna(subset=["dst_port"])
    alerts["dst_port"] = alerts["dst_port"].astype(int)
    # Parse dates
    def snort_ts_to_date(ts_str):
        try:
            month_day = ts_str.split("-")[0]
            month, day = month_day.split("/")
            return f"2017-{int(month):02d}-{int(day):02d}"
        except Exception:
            return None

    alerts["date"] = alerts["timestamp"].apply(
        snort_ts_to_date
    )
    alerts = alerts.dropna(subset=["date"])

    if "Timestamp" in flows.columns:
        flow_ts = pd.to_datetime(
            flows["Timestamp"],
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )
        flows["date"] = flow_ts.dt.strftime("%Y-%m-%d")

    print(
        f"  Snort dates: "
        f"{sorted(alerts['date'].unique())}"
    )
    print(
        f"  CIC-IDS dates: "
        f"{sorted(flows['date'].dropna().unique())}"
    )

    from collections import defaultdict

    daily_keys = defaultdict(set)

    for _, row in alerts.iterrows():
        date = row["date"]
        src = str(row["src_ip"]).strip()
        dst = str(row["dst_ip"]).strip()
        port = int(row["dst_port"])

        daily_keys[date].add((src, dst, port)) # Forward only: exact src → dst on dst_port

    total = sum(len(v) for v in daily_keys.values())
    print(f"  Unique forward keys: {total}")
    print("  Matching flows...")
    flows["Destination Port"] = pd.to_numeric(
        flows["Destination Port"], errors="coerce"
    )

    def check_flow(row):
        date = row.get("date")
        if pd.isna(date) or date not in daily_keys:
            return 0
        try:
            src = str(row["Source IP"]).strip()
            dst = str(row["Destination IP"]).strip()
            port = int(row["Destination Port"])
        except (ValueError, TypeError):
            return 0

        if (src, dst, port) in daily_keys[date]:
            return 1
        if (dst, src, port) in daily_keys[date]: # Check reverse IPs same port
            return 1
        return 0

    flows["snort_prediction"] = flows.apply(
        check_flow, axis=1
    )

    elapsed = time.time() - start
    flagged = flows["snort_prediction"].sum()
    total_flows = len(flows)
    print(f"  Flows flagged: {flagged}/{total_flows}")
    print(f"  Mapping time: {elapsed:.2f}s")
    return flows, elapsed

def evaluate_snort(flows: pd.DataFrame, mapping_time: float) -> None: # Calculate and save all Snort evaluation metrics

    y_true = flows["label_binary"].values
    y_pred = flows["snort_prediction"].values

    # CONFUSION MATRIX
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall_val = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n=== SNORT EVALUATION RESULTS ===")
    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives:  {tn}")
    print(f"False Negatives: {fn}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall_val:.4f}")
    print(f"F1 Score:        {f1:.4f}")
    print(f"FPR:             {fpr:.4f}")
    print(f"Detection Rate:  {detection_rate:.4f}")

    # SAVE METRICS (same format as anomaly)
    metrics = {
        "model": "Snort Rule-Based",
        "total_samples": int(len(y_true)),
        "attack_samples": int(y_true.sum()),
        "benign_samples": int((y_true == 0).sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "precision": round(precision, 4),
        "recall": round(recall_val, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "detection_rate": round(detection_rate, 4),
        "train_time_seconds": 0,
        "prediction_time_seconds": round(mapping_time, 2),
    }
    pd.DataFrame([metrics]).to_csv(
        TAB_DIR / "snort_metrics.csv", index=False
    )
    with open(TAB_DIR / "snort_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # PER ATTACK TYPE
    attack_perf = []
    for attack_type in flows["label_original"].unique():
        mask = flows["label_original"] == attack_type
        if mask.sum() == 0:
            continue
        a_truth = y_true[mask]
        a_preds = y_pred[mask]
        detected = int((a_preds == 1).sum())
        total = int(mask.sum())
        attack_perf.append({
            "attack_type": attack_type,
            "total": total,
            "detected_as_attack": detected,
            "detection_rate": round(
                detected / total * 100, 2
            ),
        })

    attack_df = pd.DataFrame(attack_perf).sort_values(
        "detection_rate", ascending=False
    )
    attack_df.to_csv(
        TAB_DIR / "snort_per_attack_detection.csv", index=False
    )
    print("\nPer attack type detection:")
    print(attack_df.to_string(index=False))

    # SAVE per-flow predictions (for hybrid)

    pred_df = flows[[
        "label_original",
        "label_binary",
        "snort_prediction",
    ]].copy()
    pred_df.to_csv(
        TAB_DIR / "snort_predictions.csv", index=False
    )

    # FIGURES

    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)

    labels = [[f"{val:,}" for val in row] for row in cm]
    labels = np.array(labels)

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Reds",
        xticklabels=["BENIGN", "ATTACK"],
        yticklabels=["BENIGN", "ATTACK"],
        cbar=False,
    )

    plt.title("Snort Rule-Based - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "snort_confusion_matrix.png", dpi=220
    )
    plt.close()

    # 2. Per Attack Type Detection
    plt.figure(figsize=(12, 6))
    attack_plot = attack_df[
        attack_df["attack_type"] != "BENIGN"
    ].copy()
    if len(attack_plot) > 0:
        plt.barh(
            attack_plot["attack_type"],
            attack_plot["detection_rate"],
            color="#2196F3",
        )
        plt.title("Snort: Detection Rate by Attack Type")
        plt.xlabel("Detection Rate (%)")
        plt.ylabel("Attack Type")
        plt.tight_layout()
        plt.savefig(
            FIG_DIR / "snort_per_attack_detection.png",
            dpi=220,
        )
        plt.close()

    # 3. What Snort flagged vs reality
    summary_data = {
        "Category": [
            "True Positives\n(Correct Detections)",
            "False Positives\n(False Alarms)",
            "False Negatives\n(Missed Attacks)",
            "True Negatives\n(Correct Passes)",
        ],
        "Count": [tp, fp, fn, tn],
    }
    plt.figure(figsize=(10, 6))
    colors = ["#4CAF50", "#FF9800", "#F44336", "#2196F3"]
    plt.bar(
        summary_data["Category"],
        summary_data["Count"],
        color=colors,
    )
    plt.title("Snort Detection Breakdown")
    plt.ylabel("Number of Flows")
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "snort_detection_breakdown.png", dpi=220
    )
    plt.close()

    print(f"\nAll Snort evaluation outputs saved.")


def main() -> None:
    # Check files exist
    if not SNORT_ALERTS_FULL.exists():
        raise FileNotFoundError(
            f"Missing: {SNORT_ALERTS_FULL}\n"
            f"Run the updated parse_snort.py first to "
            f"extract IPs and ports."
        )

    if not CICIDS_CSV.exists():
        raise FileNotFoundError(
            f"Missing: {CICIDS_CSV}\n"
            f"Download CIC-IDS 2017 CSV from Kaggle and "
            f"place at: {CICIDS_CSV}"
        )

    # Load both datasets
    flows = load_cicids(CICIDS_CSV)
    alerts = load_snort_alerts(SNORT_ALERTS_FULL)
    # Map and evaluate
    flows, mapping_time = map_snort_to_flows(flows, alerts)
    evaluate_snort(flows, mapping_time)

if __name__ == "__main__":
    main()
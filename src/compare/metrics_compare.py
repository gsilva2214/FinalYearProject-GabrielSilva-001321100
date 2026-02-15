"""
Comparative Evaluation: Snort vs Anomaly-Based ML
==================================================
Loads results from both independent detection tracks.
Compares against CIC-IDS 2017 ground truth.
Produces head-to-head metrics, figures, and tables.
"""

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


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    root = project_root()

    COMPARE_DIR = root / "outputs" / "compare"
    TAB_DIR = COMPARE_DIR / "tables"
    FIG_DIR = COMPARE_DIR / "figures"
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # LOAD METRICS FROM BOTH TRACKS
    # ==========================================
    anomaly_metrics_path = (
        root / "outputs" / "anomaly" / "tables"
        / "anomaly_metrics.json"
    )
    snort_metrics_path = (
        root / "outputs" / "snort" / "tables"
        / "snort_metrics.json"
    )

    if not anomaly_metrics_path.exists():
        raise FileNotFoundError(
            f"Missing: {anomaly_metrics_path}\n"
            f"Run anomaly_train_score.py first."
        )

    if not snort_metrics_path.exists():
        raise FileNotFoundError(
            f"Missing: {snort_metrics_path}\n"
            f"Run snort_evaluate.py first."
        )

    print("Loading metrics...")
    anomaly_metrics = load_json(anomaly_metrics_path)
    snort_metrics = load_json(snort_metrics_path)

    print(f"  Anomaly F1: {anomaly_metrics['f1_score']}")
    print(f"  Snort F1:   {snort_metrics['f1_score']}")

    # ==========================================
    # HEAD TO HEAD COMPARISON TABLE
    # ==========================================
    print("\n=== HEAD TO HEAD COMPARISON ===")

    comparison = {
        "Metric": [
            "Precision",
            "Recall",
            "F1 Score",
            "False Positive Rate",
            "Detection Rate",
            "True Positives",
            "False Positives",
            "True Negatives",
            "False Negatives",
            "Training Time (s)",
            "Prediction Time (s)",
        ],
        "Rule-Based (Snort)": [
            snort_metrics.get("precision", "N/A"),
            snort_metrics.get("recall", "N/A"),
            snort_metrics.get("f1_score", "N/A"),
            snort_metrics.get("false_positive_rate", "N/A"),
            snort_metrics.get("detection_rate", "N/A"),
            snort_metrics.get("true_positives", "N/A"),
            snort_metrics.get("false_positives", "N/A"),
            snort_metrics.get("true_negatives", "N/A"),
            snort_metrics.get("false_negatives", "N/A"),
            snort_metrics.get("train_time_seconds", "N/A"),
            snort_metrics.get(
                "prediction_time_seconds", "N/A"
            ),
        ],
        "Anomaly-Based (ML)": [
            anomaly_metrics["precision"],
            anomaly_metrics["recall"],
            anomaly_metrics["f1_score"],
            anomaly_metrics["false_positive_rate"],
            anomaly_metrics["detection_rate"],
            anomaly_metrics["true_positives"],
            anomaly_metrics["false_positives"],
            anomaly_metrics["true_negatives"],
            anomaly_metrics["false_negatives"],
            anomaly_metrics["train_time_seconds"],
            anomaly_metrics["prediction_time_seconds"],
        ],
    }

    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv(
        TAB_DIR / "head_to_head_comparison.csv", index=False
    )
    print(comp_df.to_string(index=False))

    # ==========================================
    # PER ATTACK TYPE COMPARISON
    # ==========================================
    anomaly_per_attack_path = (
        root / "outputs" / "anomaly" / "tables"
        / "per_attack_detection.csv"
    )
    snort_per_attack_path = (
        root / "outputs" / "snort" / "tables"
        / "snort_per_attack_detection.csv"
    )

    if (anomaly_per_attack_path.exists()
            and snort_per_attack_path.exists()):

        print("\n=== PER ATTACK TYPE COMPARISON ===")

        anom_attacks = pd.read_csv(anomaly_per_attack_path)
        anom_attacks = anom_attacks.rename(columns={
            "detection_rate": "ml_detection_rate",
            "detected_as_anomaly": "ml_detected",
        })

        snort_attacks = pd.read_csv(snort_per_attack_path)
        snort_attacks = snort_attacks.rename(columns={
            "detection_rate": "snort_detection_rate",
            "detected_as_attack": "snort_detected",
        })

        per_attack = anom_attacks.merge(
            snort_attacks,
            on="attack_type",
            how="outer",
            suffixes=("_ml", "_snort"),
        )

        per_attack.to_csv(
            TAB_DIR / "per_attack_comparison.csv",
            index=False,
        )
        print(per_attack.to_string(index=False))

        # ---- Figure: Side by side bar chart ----
        plot_df = per_attack[
            per_attack["attack_type"] != "BENIGN"
        ].copy()

        if len(plot_df) > 0:
            fig, ax = plt.subplots(figsize=(14, 7))
            x = np.arange(len(plot_df))
            width = 0.35

            snort_vals = plot_df[
                "snort_detection_rate"
            ].fillna(0).values
            ml_vals = plot_df[
                "ml_detection_rate"
            ].fillna(0).values

            ax.bar(
                x - width / 2,
                snort_vals,
                width,
                label="Snort (Rule-Based)",
                color="#2196F3",
            )
            ax.bar(
                x + width / 2,
                ml_vals,
                width,
                label="ML (Anomaly-Based)",
                color="#FF9800",
            )

            ax.set_ylabel("Detection Rate (%)")
            ax.set_title(
                "Detection Rate by Attack Type: "
                "Snort vs ML"
            )
            ax.set_xticks(x)
            ax.set_xticklabels(
                plot_df["attack_type"],
                rotation=45,
                ha="right",
            )
            ax.legend()
            plt.tight_layout()
            plt.savefig(
                FIG_DIR / "01_per_attack_comparison.png",
                dpi=220,
            )
            plt.close()
            print("\nSaved: 01_per_attack_comparison.png")

    # ==========================================
    # OVERALL METRICS BAR CHART
    # ==========================================
    metrics_to_plot = [
        "precision", "recall", "f1_score",
        "false_positive_rate",
    ]
    snort_vals = [
        float(snort_metrics.get(m, 0))
        for m in metrics_to_plot
    ]
    ml_vals = [
        float(anomaly_metrics.get(m, 0))
        for m in metrics_to_plot
    ]
    labels = ["Precision", "Recall", "F1 Score", "FPR"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(
        x - width / 2, snort_vals, width,
        label="Snort (Rule-Based)", color="#2196F3",
    )
    ax.bar(
        x + width / 2, ml_vals, width,
        label="ML (Anomaly-Based)", color="#FF9800",
    )

    ax.set_ylabel("Score")
    ax.set_title("Overall Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "02_overall_comparison.png", dpi=220
    )
    plt.close()
    print("Saved: 02_overall_comparison.png")

    # ==========================================
    # HYBRID ANALYSIS
    # ==========================================
    anomaly_pred_path = (
        root / "outputs" / "anomaly" / "tables"
        / "anomaly_predictions.csv"
    )
    snort_pred_path = (
        root / "outputs" / "snort" / "tables"
        / "snort_predictions.csv"
    )

    if anomaly_pred_path.exists() and snort_pred_path.exists():
        print("\n=== HYBRID ANALYSIS ===")

        anom_preds = pd.read_csv(anomaly_pred_path)
        snort_preds = pd.read_csv(snort_pred_path)

        # Check they are same length
        if len(anom_preds) != len(snort_preds):
            print(
                f"WARNING: Different lengths. "
                f"Anomaly: {len(anom_preds)}, "
                f"Snort: {len(snort_preds)}"
            )
            # Use the shorter one
            min_len = min(len(anom_preds), len(snort_preds))
            anom_preds = anom_preds.iloc[:min_len]
            snort_preds = snort_preds.iloc[:min_len]

        y_true = anom_preds["label_binary"].values
        y_ml = anom_preds["ml_prediction"].values
        y_snort = snort_preds["snort_prediction"].values

        # OR: flag if EITHER method flags it
        y_hybrid_or = np.where(
            (y_ml == 1) | (y_snort == 1), 1, 0
        )

        # AND: flag only if BOTH flag it
        y_hybrid_and = np.where(
            (y_ml == 1) & (y_snort == 1), 1, 0
        )

        hybrid_results = []
        for name, preds in [
            ("Snort Only", y_snort),
            ("ML Only", y_ml),
            ("Hybrid OR", y_hybrid_or),
            ("Hybrid AND", y_hybrid_and),
        ]:
            tn, fp, fn, tp = confusion_matrix(
                y_true, preds
            ).ravel()
            hybrid_results.append({
                "Method": name,
                "Precision": round(
                    precision_score(
                        y_true, preds, zero_division=0
                    ), 4
                ),
                "Recall": round(
                    recall_score(
                        y_true, preds, zero_division=0
                    ), 4
                ),
                "F1": round(
                    f1_score(
                        y_true, preds, zero_division=0
                    ), 4
                ),
                "FPR": round(
                    fp / (fp + tn) if (fp + tn) > 0
                    else 0, 4
                ),
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn),
            })

        hybrid_df = pd.DataFrame(hybrid_results)
        hybrid_df.to_csv(
            TAB_DIR / "hybrid_comparison.csv", index=False
        )
        print(hybrid_df.to_string(index=False))

        # ---- Figure: Hybrid comparison ----
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(hybrid_df))
        width = 0.2

        ax.bar(
            x - width * 1.5,
            hybrid_df["Precision"],
            width, label="Precision",
        )
        ax.bar(
            x - width * 0.5,
            hybrid_df["Recall"],
            width, label="Recall",
        )
        ax.bar(
            x + width * 0.5,
            hybrid_df["F1"],
            width, label="F1",
        )
        ax.bar(
            x + width * 1.5,
            hybrid_df["FPR"],
            width, label="FPR",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(hybrid_df["Method"])
        ax.set_title(
            "Hybrid Analysis: All Methods Compared"
        )
        ax.set_ylabel("Score")
        ax.legend()
        ax.set_ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(
            FIG_DIR / "03_hybrid_comparison.png", dpi=220
        )
        plt.close()
        print("Saved: 03_hybrid_comparison.png")

    else:
        print(
            "Skipping hybrid analysis. "
            "Missing prediction files."
        )
        if not anomaly_pred_path.exists():
            print(f"  Missing: {anomaly_pred_path}")
        if not snort_pred_path.exists():
            print(f"  Missing: {snort_pred_path}")

    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    summary = {
        "anomaly_model": anomaly_metrics.get("model"),
        "anomaly_f1": anomaly_metrics.get("f1_score"),
        "anomaly_fpr": anomaly_metrics.get(
            "false_positive_rate"
        ),
        "snort_f1": snort_metrics.get("f1_score"),
        "snort_fpr": snort_metrics.get(
            "false_positive_rate"
        ),
    }

    if (anomaly_metrics.get("f1_score", 0)
            > snort_metrics.get("f1_score", 0)):
        summary["better_f1"] = "Anomaly-Based"
    else:
        summary["better_f1"] = "Rule-Based"

    if (anomaly_metrics.get("false_positive_rate", 1)
            < snort_metrics.get("false_positive_rate", 1)):
        summary["lower_fpr"] = "Anomaly-Based"
    else:
        summary["lower_fpr"] = "Rule-Based"

    pd.DataFrame([summary]).to_csv(
        TAB_DIR / "final_summary.csv", index=False
    )

    with open(TAB_DIR / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== FINAL SUMMARY ===")
    print(f"Better F1:  {summary['better_f1']}")
    print(f"Lower FPR:  {summary['lower_fpr']}")

    print(f"\nAll outputs saved to:")
    print(f"  Tables:  {TAB_DIR}")
    print(f"  Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
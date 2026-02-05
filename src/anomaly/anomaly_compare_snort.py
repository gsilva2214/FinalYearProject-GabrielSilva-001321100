from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ANOM_DIR = ROOT / "outputs" / "anomaly"
TAB_DIR = ANOM_DIR / "tables"
FIG_DIR = ANOM_DIR / "figures"
TAB_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SECONDS = 60
SCORES_CSV = TAB_DIR / f"anomaly_scores_{WINDOW_SECONDS}s.csv"

LOW_SNORT_ALERTS_CUTOFF = 0  # anomaly window with 0 snort alerts in that window


def save_fig(path: Path, title: str, xlab: str, ylab: str) -> None:
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main() -> None:
    if not SCORES_CSV.exists():
        raise FileNotFoundError(f"Missing: {SCORES_CSV}. Run anomaly_train_score.py first.")

    df = pd.read_csv(SCORES_CSV)
    df["window_start"] = pd.to_datetime(df["window_start"], errors="coerce")
    df = df.dropna(subset=["window_start"]).copy()
    df = df.sort_values("window_start")

    if "alerts" not in df.columns:
        raise KeyError("Missing alerts feature in scored file. Rebuild features.")

    plt.figure(figsize=(10, 4))
    plt.plot(df["window_start"], df["alerts"])
    save_fig(FIG_DIR / "03_snort_alert_volume_per_window.png", "Snort alert volume per window", "Time", "Alerts")

    plt.figure(figsize=(10, 4))
    plt.plot(df["window_start"], df["anomaly_score"], label="Anomaly score")
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(df["window_start"], df["alerts"], linestyle="--", label="Alerts")
    plt.title("Anomaly score vs Snort alert volume")
    ax.set_xlabel("Time")
    ax.set_ylabel("Anomaly score")
    ax2.set_ylabel("Alerts")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_anomaly_score_vs_snort_alerts.png", dpi=220)
    plt.close()

    anomalous = df[df["is_anomaly"] == 1].copy()
    anomalous_low_snort = anomalous[anomalous["alerts"] <= LOW_SNORT_ALERTS_CUTOFF].copy()

    out1 = TAB_DIR / f"anomaly_windows_{WINDOW_SECONDS}s.csv"
    out2 = TAB_DIR / f"anomaly_windows_low_snort_{WINDOW_SECONDS}s.csv"
    anomalous.to_csv(out1, index=False)
    anomalous_low_snort.to_csv(out2, index=False)

    summary = {
        "window_seconds": WINDOW_SECONDS,
        "anomaly_windows": len(anomalous),
        "anomaly_windows_low_snort": len(anomalous_low_snort),
        "low_snort_cutoff": LOW_SNORT_ALERTS_CUTOFF,
        "pct_anomaly_low_snort": (100.0 * len(anomalous_low_snort) / len(anomalous)) if len(anomalous) else 0.0,
    }
    pd.DataFrame([summary]).to_csv(TAB_DIR / f"compare_summary_{WINDOW_SECONDS}s.csv", index=False)

    print("Wrote:", out1)
    print("Wrote:", out2)
    print("Anomaly windows:", len(anomalous))
    print("Anomaly windows with low Snort alerts:", len(anomalous_low_snort))


if __name__ == "__main__":
    main()

from pathlib import Path
import pandas as pd


WINDOW_SECONDS = 60

# rule for snort flagged this window
SNORT_ALERT_THRESHOLD = 1  # alerts >= 1 means flagged


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def confusion(y_true: pd.Series, y_pred: pd.Series) -> tuple[int, int, int, int]:
    yt = y_true.astype(int)
    yp = y_pred.astype(int)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    return tp, fp, tn, fn


def main() -> None:
    root = project_root()

    scores_csv = root / "outputs" / "anomaly" / "tables" / f"anomaly_scores_{WINDOW_SECONDS}s.csv"
    if not scores_csv.exists():
        raise FileNotFoundError(f"Missing: {scores_csv}")

    df = pd.read_csv(scores_csv)

    needed = {"window_start", "alerts", "is_anomaly", "anomaly_score"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {scores_csv.name}: {missing}")

    df["snort_pred"] = (df["alerts"].astype(float) >= SNORT_ALERT_THRESHOLD).astype(int)
    df["anomaly_pred"] = df["is_anomaly"].astype(int)

    n = len(df)
    snort_rate = 100.0 * df["snort_pred"].mean()
    anom_rate = 100.0 * df["anomaly_pred"].mean()


    both = int(((df["snort_pred"] == 1) & (df["anomaly_pred"] == 1)).sum())
    snort_only = int(((df["snort_pred"] == 1) & (df["anomaly_pred"] == 0)).sum())
    anom_only = int(((df["snort_pred"] == 0) & (df["anomaly_pred"] == 1)).sum())
    neither = int(((df["snort_pred"] == 0) & (df["anomaly_pred"] == 0)).sum())
    union = both + snort_only + anom_only
    jaccard = safe_div(both, union)


    tp_a, fp_a, tn_a, fn_a = confusion(df["snort_pred"], df["anomaly_pred"])
    p_a, r_a, f1_a = prf(tp_a, fp_a, fn_a)

    # Treat anomaly as reference label, then score Snort against it
    tp_s, fp_s, tn_s, fn_s = confusion(df["anomaly_pred"], df["snort_pred"])
    p_s, r_s, f1_s = prf(tp_s, fp_s, fn_s)

    out_dir = root / "outputs" / "compare" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "window_seconds": WINDOW_SECONDS,
        "snort_alert_threshold": SNORT_ALERT_THRESHOLD,
        "windows_total": n,
        "snort_flagged_pct": snort_rate,
        "anomaly_flagged_pct": anom_rate,
        "both_flagged": both,
        "snort_only": snort_only,
        "anomaly_only": anom_only,
        "neither": neither,
        "jaccard_overlap": jaccard,
        "anomaly_vs_snort_precision": p_a,
        "anomaly_vs_snort_recall": r_a,
        "anomaly_vs_snort_f1": f1_a,
        "snort_vs_anomaly_precision": p_s,
        "snort_vs_anomaly_recall": r_s,
        "snort_vs_anomaly_f1": f1_s,
    }

    pd.DataFrame([summary]).to_csv(out_dir / f"metrics_compare_{WINDOW_SECONDS}s.csv", index=False)

    # Detailed rows for appendix or later plots
    keep = ["window_start", "alerts", "snort_pred", "anomaly_pred", "anomaly_score"]
    df[keep].to_csv(out_dir / f"window_level_flags_{WINDOW_SECONDS}s.csv", index=False)

    print("Wrote:", out_dir / f"metrics_compare_{WINDOW_SECONDS}s.csv")
    print("Wrote:", out_dir / f"window_level_flags_{WINDOW_SECONDS}s.csv")


if __name__ == "__main__":
    main()

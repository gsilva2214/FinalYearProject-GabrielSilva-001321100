from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[2]
TAB_DIR = ROOT / "outputs" / "anomaly" / "tables"
FIG_DIR = ROOT / "outputs" / "anomaly" / "figures"
TAB_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SECONDS = 60
FEATURES_CSV = TAB_DIR / f"anomaly_features_{WINDOW_SECONDS}s.csv"

RANDOM_SEED = 42
CONTAMINATION = "auto"

THRESHOLD_PERCENTILE = 99.0  # anomalies are top 1 percent of baseline scores


def pick_feature_cols(df: pd.DataFrame) -> list[str]:
    ignore = {"window_start", "day_name", "date"}
    cols = [c for c in df.columns if c not in ignore]

    bad = []
    for c in cols:
        if df[c].dtype == "O":
            bad.append(c)
    cols = [c for c in cols if c not in bad]

    keep = [c for c in cols if c not in ["hour"]]
    keep.append("hour")
    return keep


def save_fig(path: Path, title: str, xlab: str, ylab: str) -> None:
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def choose_baseline_day(df: pd.DataFrame) -> str:
    per_day = df.groupby("day_name")["alerts"].sum().sort_values()
    per_day = per_day[per_day > 0]
    if len(per_day) == 0:
        raise RuntimeError("No alerts in features. Check earlier steps.")
    return str(per_day.index[0])


def main() -> None:
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Missing: {FEATURES_CSV}. Run anomaly_features.py first.")

    df = pd.read_csv(FEATURES_CSV)
    df["window_start"] = pd.to_datetime(df["window_start"], errors="coerce")
    df = df.dropna(subset=["window_start"]).copy()
    df = df.sort_values("window_start")

    baseline_day = choose_baseline_day(df)

    feat_cols = pick_feature_cols(df)
    X = df[feat_cols].fillna(0).astype(float)

    base_mask = df["day_name"] == baseline_day
    X_base = X.loc[base_mask].copy()

    if len(X_base) < 30:
        raise RuntimeError("Baseline too small. Increase window size or pick a different baseline day.")

    model = IsolationForest(
        n_estimators=300,
        random_state=RANDOM_SEED,
        contamination=CONTAMINATION,
    )
    model.fit(X_base)

    score = -model.decision_function(X)
    df["anomaly_score"] = score

    base_scores = df.loc[base_mask, "anomaly_score"].values
    thresh = float(np.percentile(base_scores, THRESHOLD_PERCENTILE))
    df["is_anomaly"] = (df["anomaly_score"] >= thresh).astype(int)

    out_scores = TAB_DIR / f"anomaly_scores_{WINDOW_SECONDS}s.csv"
    df.to_csv(out_scores, index=False)

    meta = {
        "window_seconds": WINDOW_SECONDS,
        "baseline_day": baseline_day,
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "threshold_value": thresh,
        "feature_count": len(feat_cols),
        "baseline_rows": int(base_mask.sum()),
        "total_rows": len(df),
    }
    meta_path = TAB_DIR / f"anomaly_model_meta_{WINDOW_SECONDS}s.csv"
    pd.DataFrame([meta]).to_csv(meta_path, index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(df["window_start"], df["anomaly_score"])
    save_fig(FIG_DIR / "01_anomaly_score_over_time.png", "Anomaly score over time", "Time", "Anomaly score")

    per_day = df.groupby("date")["is_anomaly"].mean().rename("pct_anomalous_windows").reset_index()
    per_day["pct_anomalous_windows"] = per_day["pct_anomalous_windows"] * 100.0
    per_day.to_csv(TAB_DIR / f"anomaly_percent_per_day_{WINDOW_SECONDS}s.csv", index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(per_day["date"], per_day["pct_anomalous_windows"], marker="o")
    plt.xticks(rotation=45, ha="right")
    save_fig(FIG_DIR / "02_percent_anomalous_windows_per_day.png", "Percent anomalous windows per day", "Date", "Percent")

    print("Wrote:", out_scores)
    print("Wrote:", meta_path)
    print("Baseline day:", baseline_day)
    print("Threshold value:", thresh)


if __name__ == "__main__":
    main()

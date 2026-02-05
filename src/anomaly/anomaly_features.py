from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN_CSV = ROOT / "data" / "snort_alerts.csv"
OUT_DIR = ROOT / "outputs" / "anomaly"
TAB_DIR = OUT_DIR / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)

CAPTURE_YEAR = "2024"
TS_WITH_YEAR_FMT = "%Y/%m/%d/%H:%M:%S.%f"

WINDOW_SECONDS = 60  # set 10, 30, 60 depending on how bursty your data is


def pick_col(df: pd.DataFrame, names: list[str]) -> str:
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    for c in cols:
        lc = c.lower()
        for n in names:
            if n.lower() in lc:
                return c
    raise KeyError(f"Missing column. Tried {names}. Present {cols}")


def parse_snort_ts(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = CAPTURE_YEAR + "/" + s.str.replace("-", "/", n=1)
    out = pd.to_datetime(s, format=TS_WITH_YEAR_FMT, errors="coerce")
    if out.isna().mean() > 0.50:
        out = pd.to_datetime(s, errors="coerce")
    return out


def entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def main() -> None:
    df = pd.read_csv(IN_CSV)

    ts_col = pick_col(df, ["timestamp", "time", "datetime", "event_time", "alert_time", "ts"])
    sid_col = pick_col(df, ["sid", "signature_id", "sig_id", "rule_id"])
    prio_col = pick_col(df, ["priority", "prio", "alert_priority", "severity", "sev"])

    df[ts_col] = parse_snort_ts(df[ts_col])
    df = df.dropna(subset=[ts_col]).copy()

    df[sid_col] = pd.to_numeric(df[sid_col], errors="coerce")
    df[prio_col] = pd.to_numeric(df[prio_col], errors="coerce")
    df = df.dropna(subset=[sid_col, prio_col]).copy()
    df[sid_col] = df[sid_col].astype(int)
    df[prio_col] = df[prio_col].astype(int)

    df = df.sort_values(ts_col).copy()
    df["day_name"] = df[ts_col].dt.day_name()
    df["date"] = df[ts_col].dt.date.astype(str)

    df = df.set_index(ts_col)

    rule_share = df.groupby(pd.Grouper(freq=f"{WINDOW_SECONDS}S"))[sid_col].value_counts(normalize=True)
    top_rule_share = rule_share.groupby(level=0).max().rename("top_rule_share")

    prio_counts = (
        df.groupby(pd.Grouper(freq=f"{WINDOW_SECONDS}S"))[prio_col]
        .value_counts()
        .unstack(fill_value=0)
        .add_prefix("prio_")
    )

    base = df.groupby(pd.Grouper(freq=f"{WINDOW_SECONDS}S")).agg(
        alerts=("date", "size"),
        unique_sids=(sid_col, "nunique"),
        unique_priorities=(prio_col, "nunique"),
    )

    sid_ent = []
    idx = []
    for t, g in df.groupby(pd.Grouper(freq=f"{WINDOW_SECONDS}S")):
        idx.append(t)
        sid_counts = g[sid_col].value_counts().values.astype(float)
        sid_ent.append(entropy_from_counts(sid_counts))
    sid_ent = pd.Series(sid_ent, index=idx, name="sid_entropy")

    feat = base.join(prio_counts, how="left").join(top_rule_share, how="left").join(sid_ent, how="left")
    feat = feat.fillna(0)

    feat["hour"] = feat.index.hour.astype(int)
    feat["day_name"] = feat.index.day_name()
    feat["date"] = feat.index.date.astype(str)

    out_path = TAB_DIR / f"anomaly_features_{WINDOW_SECONDS}s.csv"
    feat.reset_index(names="window_start").to_csv(out_path, index=False)

    print("Wrote:", out_path)
    print("Rows:", len(feat))
    print("Window seconds:", WINDOW_SECONDS)


if __name__ == "__main__":
    main()

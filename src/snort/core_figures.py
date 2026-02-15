from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
IN_CSV = ROOT / "outputs" / "snort" / "snort_alerts.csv"
FIG_DIR = ROOT / "outputs" / "snort" / "figures"
TAB_DIR = ROOT / "outputs" / "snort" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

CAPTURE_YEAR = "2017"

TS_FORMAT_WITH_YEAR = "%Y/%m/%d/%H:%M:%S.%f"


def pick_col(df: pd.DataFrame, names: list[str], required: bool = True) -> str | None:
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

    if required:
        raise KeyError(f"Missing column. Tried {names}. Present {cols}")
    return None

def parse_snort_ts(series: pd.Series) -> pd.Series:

    s = series.astype(str)
    s = CAPTURE_YEAR + "/" + s.str.replace("-", "/", n=1)
    out = pd.to_datetime(s, format=TS_FORMAT_WITH_YEAR, errors="coerce")
    if out.isna().mean() > 0.50:
        out = pd.to_datetime(s, errors="coerce")
    return out

def save_fig(path: Path, title: str, xlab: str, ylab: str) -> None:
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()

def load_df() -> tuple[pd.DataFrame, str, str, str]:
    df = pd.read_csv(IN_CSV)

    ts_col = pick_col(df, ["timestamp", "time", "datetime", "event_time", "alert_time", "ts"])
    sid_col = pick_col(df, ["sid", "signature_id", "sig_id", "rule_id"])
    prio_col = pick_col(df, ["priority", "prio", "alert_priority", "severity", "sev"])
    df[ts_col] = parse_snort_ts(df[ts_col])
    df = df.dropna(subset=[ts_col]).copy()
    df["date"] = df[ts_col].dt.date.astype(str)
    df["hour"] = df[ts_col].dt.hour.astype(int)

    return df, ts_col, sid_col, prio_col

def alerts_per_day(df: pd.DataFrame) -> None:
    t = df.groupby("date").size().sort_index().rename("alerts").reset_index()

    plt.figure(figsize=(10, 4))
    plt.plot(t["date"], t["alerts"], marker="o")
    plt.xticks(rotation=45, ha="right")
    save_fig(FIG_DIR / "01_alerts_per_day.png", "Alerts per day", "Date", "Alerts")
    t.to_csv(TAB_DIR / "alerts_per_day.csv", index=False)

def alerts_per_hour(df: pd.DataFrame) -> None:
    t = df.groupby("hour").size().reindex(range(24), fill_value=0).rename("alerts").reset_index()
    plt.figure(figsize=(10, 4))
    plt.bar(t["hour"], t["alerts"])
    plt.xticks(range(24))
    save_fig(FIG_DIR / "02_alerts_per_hour.png", "Alerts per hour", "Hour", "Alerts")
    t.to_csv(TAB_DIR / "alerts_per_hour.csv", index=False)

def top10_signatures(df: pd.DataFrame, sid_col: str) -> None:
    t = df[sid_col].value_counts().head(10).rename_axis("sid").reset_index(name="alerts")
    plt.figure(figsize=(10, 5))
    plt.bar(t["sid"].astype(str), t["alerts"])
    plt.xticks(rotation=45, ha="right")
    save_fig(FIG_DIR / "03_top10_signatures_overall.png", "Top 10 signatures overall", "SID", "Alerts")

    t.to_csv(TAB_DIR / "top10_signatures.csv", index=False)

def priority_distribution(df: pd.DataFrame, prio_col: str) -> None:
    t = df[prio_col].astype(str).value_counts().sort_index().rename_axis("priority").reset_index(name="alerts")

    plt.figure(figsize=(8, 4))
    plt.bar(t["priority"], t["alerts"])
    save_fig(FIG_DIR / "04_priority_distribution.png", "Priority distribution", "Priority", "Alerts")
    t.to_csv(TAB_DIR / "priority_distribution.csv", index=False)

def main() -> None:
    df, ts_col, sid_col, prio_col = load_df()
    alerts_per_day(df)
    alerts_per_hour(df)
    top10_signatures(df, sid_col)
    priority_distribution(df, prio_col)
    print("Core figures saved in outputs/snort/figures")
    print("Tables saved in outputs/snort/tables")
    print("Columns:", ts_col, sid_col, prio_col)

if __name__ == "__main__":
    main()
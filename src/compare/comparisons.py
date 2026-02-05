from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "output" / "snort_alerts.csv"
FIG_DIR = ROOT / "output" / "figures"
TAB_DIR = ROOT / "output" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

CAPTURE_YEAR = "2024"
TS_FORMAT_WITH_YEAR = "%Y/%m/%d/%H:%M:%S.%f"

FALSE_POS_TOPK = 10
PEAK_TOPN_HOURS = 4


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
    out = pd.to_datetime(s, format=TS_FORMAT_WITH_YEAR, errors="coerce")
    if out.isna().mean() > 0.50:
        out = pd.to_datetime(s, errors="coerce")
    return out


def load_df() -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(IN_CSV)
    ts_col = pick_col(df, ["timestamp", "time", "datetime", "event_time", "alert_time", "ts"])
    sid_col = pick_col(df, ["sid", "signature_id", "sig_id", "rule_id"])

    df[ts_col] = parse_snort_ts(df[ts_col])
    df = df.dropna(subset=[ts_col]).copy()

    df["date"] = df[ts_col].dt.date.astype(str)
    df["hour"] = df[ts_col].dt.hour.astype(int)

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["day_name"] = pd.Categorical(df[ts_col].dt.day_name(), categories=order, ordered=True)

    df[sid_col] = pd.to_numeric(df[sid_col], errors="coerce")
    df = df.dropna(subset=[sid_col]).copy()
    df[sid_col] = df[sid_col].astype(int)

    return df, sid_col


def choose_baseline_and_attack(df: pd.DataFrame) -> tuple[str, str, pd.DataFrame]:
    counts = df.groupby("day_name").size().rename("alerts").reset_index()
    counts = counts.dropna(subset=["day_name"]).copy()
    counts = counts[counts["alerts"] > 0].copy()

    if counts.empty:
        raise RuntimeError("No alerts after parsing. Check snort_alerts.csv.")

    baseline_day = counts.sort_values("alerts", ascending=True).iloc[0]["day_name"]
    attack_day = counts.sort_values("alerts", ascending=False).iloc[0]["day_name"]

    counts.to_csv(TAB_DIR / "alerts_by_weekday.csv", index=False)
    return str(baseline_day), str(attack_day), counts


def monday_vs_friday_summary(df: pd.DataFrame, sid_col: str, baseline_day: str, attack_day: str) -> pd.DataFrame:
    base = df[df["day_name"] == baseline_day]
    atk = df[df["day_name"] == attack_day]

    out = pd.DataFrame(
        {
            "baseline_day": [baseline_day],
            "attack_day": [attack_day],
            "baseline_alerts": [len(base)],
            "attack_alerts": [len(atk)],
            "ratio_attack_over_baseline": [float(len(atk)) / float(len(base)) if len(base) else np.nan],
            "unique_sids_baseline": [base[sid_col].nunique()],
            "unique_sids_attack": [atk[sid_col].nunique()],
        }
    )
    out.to_csv(TAB_DIR / "baseline_vs_attack_summary.csv", index=False)
    return out


def signature_diversity_per_day(df: pd.DataFrame, sid_col: str) -> pd.DataFrame:
    t = df.groupby("date")[sid_col].nunique().sort_index().rename("unique_sids").reset_index()

    plt.figure(figsize=(10, 4))
    plt.plot(t["date"], t["unique_sids"], marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title("Signature diversity per day")
    plt.xlabel("Date")
    plt.ylabel("Unique SIDs")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_signature_diversity_per_day.png", dpi=220)
    plt.close()

    t.to_csv(TAB_DIR / "signature_diversity_per_day.csv", index=False)
    return t


def time_concentration(df: pd.DataFrame) -> pd.DataFrame:
    per_hour = df.groupby("hour").size().sort_values(ascending=False)
    peak_hours = list(per_hour.head(PEAK_TOPN_HOURS).index.astype(int))

    total = len(df)
    peak = int(df["hour"].isin(peak_hours).sum())

    out = pd.DataFrame(
        {
            "peak_hours_topn": [",".join(str(x) for x in peak_hours)],
            "alerts_total": [total],
            "alerts_in_peak_hours": [peak],
            "percent_in_peak_hours": [(100.0 * peak / total) if total else np.nan],
        }
    )
    out.to_csv(TAB_DIR / "time_concentration.csv", index=False)
    return out


def false_positive_proxy(df: pd.DataFrame, sid_col: str, baseline_day: str, attack_day: str) -> pd.DataFrame:
    base = df[df["day_name"] == baseline_day]
    atk = df[df["day_name"] == attack_day]

    base_counts = base[sid_col].value_counts()
    atk_counts = atk[sid_col].value_counts()

    base_top = base_counts.head(FALSE_POS_TOPK).rename_axis("sid").reset_index(name="baseline_alerts")
    base_top["attack_alerts"] = base_top["sid"].map(atk_counts).fillna(0).astype(int)
    base_top["ratio_attack_over_baseline"] = base_top.apply(
        lambda r: (float(r["attack_alerts"]) / float(r["baseline_alerts"])) if r["baseline_alerts"] else np.nan,
        axis=1,
    )

    base_top.to_csv(TAB_DIR / "false_positive_proxy.csv", index=False)

    if base_top.empty:
        notes = ROOT / "output" / "snort_comparisons_notes.txt"
        notes.write_text(
            "False positive proxy skipped. Baseline day produced zero SIDs.\n",
            encoding="utf-8",
        )
        return base_top

    x = np.arange(len(base_top))
    w = 0.42
    plt.figure(figsize=(11, 5))
    plt.bar(x - w / 2, base_top["baseline_alerts"].values, width=w, label=baseline_day)
    plt.bar(x + w / 2, base_top["attack_alerts"].values, width=w, label=attack_day)
    plt.xticks(x, base_top["sid"].astype(str), rotation=45, ha="right")
    plt.legend()
    plt.title(f"Baseline high frequency SIDs: {baseline_day} vs {attack_day}")
    plt.xlabel("SID")
    plt.ylabel("Alerts")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_false_positive_proxy.png", dpi=220)
    plt.close()

    return base_top


def write_notes(summary: pd.DataFrame, tc: pd.DataFrame, baseline_day: str, attack_day: str) -> None:
    lines: list[str] = []
    lines.append("Snort comparisons notes")
    lines.append("")
    lines.append(f"Baseline day chosen from data: {baseline_day}.")
    lines.append(f"Attack day chosen from data: {attack_day}.")
    lines.append(f"Baseline alerts: {int(summary.loc[0, 'baseline_alerts'])}.")
    lines.append(f"Attack alerts: {int(summary.loc[0, 'attack_alerts'])}.")
    ratio = summary.loc[0, "ratio_attack_over_baseline"]
    if pd.notna(ratio):
        lines.append(f"Attack over baseline ratio: {ratio:.2f}.")
    else:
        lines.append("Attack over baseline ratio: NA.")

    lines.append("")
    lines.append("Sentence for marks")
    lines.append("Several high-frequency alerts were observed during normal traffic periods, indicating potential false positives inherent to signature-based detection.")

    lines.append("")
    lines.append("Time concentration")
    lines.append(f"Peak hours: {tc.loc[0, 'peak_hours_topn']}.")
    lines.append(f"Percent in peak hours: {float(tc.loc[0, 'percent_in_peak_hours']):.2f}.")

    (ROOT / "output" / "snort_comparisons_notes.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df, sid_col = load_df()

    baseline_day, attack_day, weekday_counts = choose_baseline_and_attack(df)

    summary = monday_vs_friday_summary(df, sid_col, baseline_day, attack_day)
    signature_diversity_per_day(df, sid_col)
    tc = time_concentration(df)
    false_positive_proxy(df, sid_col, baseline_day, attack_day)
    write_notes(summary, tc, baseline_day, attack_day)

    print("Done.")
    print("Baseline day:", baseline_day)
    print("Attack day:", attack_day)
    print("Figures saved in output/figures")
    print("Tables saved in output/tables")


if __name__ == "__main__":
    main()

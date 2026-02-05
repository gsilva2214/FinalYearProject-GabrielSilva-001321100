from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
IN_CSV = ROOT / "outputs" / "snort" / "snort_alerts.csv"
FIG_DIR = ROOT / "outputs" / "snort" / "figures"
TAB_DIR = ROOT / "outputs" / "snort" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

CAPTURE_YEAR = "2024"
TS_FORMAT_WITH_YEAR = "%Y/%m/%d/%H:%M:%S.%f"

ATTACK_DAY = "Friday"


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


def load_df() -> pd.DataFrame:
    df = pd.read_csv(IN_CSV)
    ts_col = pick_col(df, ["timestamp", "time", "datetime", "event_time", "alert_time", "ts"])
    df[ts_col] = parse_snort_ts(df[ts_col])
    df = df.dropna(subset=[ts_col]).copy()

    df["hour"] = df[ts_col].dt.hour.astype(int)
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["day_name"] = pd.Categorical(df[ts_col].dt.day_name(), categories=order, ordered=True)

    df["_ts"] = df[ts_col]
    return df


def heatmap_alerts_hour_day(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(index="day_name", columns="hour", values="_ts", aggfunc="size", fill_value=0)

    plt.figure(figsize=(12, 4))
    plt.imshow(pivot.values, aspect="auto")
    plt.yticks(range(pivot.shape[0]), [str(x) for x in pivot.index])
    plt.xticks(range(24), list(range(24)))
    plt.colorbar(label="Alerts")
    plt.title("Heatmap: alerts per hour per day")
    plt.xlabel("Hour")
    plt.ylabel("Day")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_heatmap_alerts_hour_day.png", dpi=220)
    plt.close()

    pivot.reset_index().to_csv(TAB_DIR / "heatmap_alerts_hour_day.csv", index=False)


def cumulative_alerts_over_time(df: pd.DataFrame) -> None:
    s = df.sort_values("_ts").copy()
    s["cum_alerts"] = np.arange(1, len(s) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(s["_ts"], s["cum_alerts"])
    plt.title("Cumulative alerts over time")
    plt.xlabel("Time")
    plt.ylabel("Cumulative alerts")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_cumulative_alerts_over_time.png", dpi=220)
    plt.close()

    s[["_ts", "cum_alerts"]].to_csv(TAB_DIR / "cumulative_alerts.csv", index=False)


def stacked_priorities_per_day(df: pd.DataFrame) -> None:
    if "priority" not in df.columns:
        return

    df2 = df.copy()
    df2["date"] = df2["_ts"].dt.date.astype(str)

    tab = df2.pivot_table(index="date", columns=df2["priority"].astype(str), values="_ts", aggfunc="size", fill_value=0).sort_index()

    plt.figure(figsize=(11, 5))
    bottoms = np.zeros(len(tab))
    for col in tab.columns:
        vals = tab[col].values
        plt.bar(tab.index.astype(str), vals, bottom=bottoms, label=str(col))
        bottoms = bottoms + vals

    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Priority", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.title("Priorities per day")
    plt.xlabel("Date")
    plt.ylabel("Alerts")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "09_stacked_priorities_per_day.png", dpi=220)
    plt.close()

    tab.reset_index().to_csv(TAB_DIR / "priorities_per_day.csv", index=False)


def top_source_ips_attack_day(df: pd.DataFrame) -> None:
    if "src_ip" not in df.columns:
        return

    fri = df[df["day_name"] == ATTACK_DAY].copy()
    if len(fri) == 0:
        return

    t = fri["src_ip"].astype(str).value_counts().head(10).rename_axis("src_ip").reset_index(name="alerts")

    plt.figure(figsize=(10, 5))
    plt.bar(t["src_ip"], t["alerts"])
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top source IPs on {ATTACK_DAY}")
    plt.xlabel("Source IP")
    plt.ylabel("Alerts")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "10_top_source_ips_friday.png", dpi=220)
    plt.close()

    t.to_csv(TAB_DIR / "top_source_ips_friday.csv", index=False)


def main() -> None:
    df = load_df()

    # Two strong extras
    heatmap_alerts_hour_day(df)
    cumulative_alerts_over_time(df)

    # Optional extras, keep if you want more figures
    stacked_priorities_per_day(df)
    top_source_ips_attack_day(df)

    print("Extras saved in outputs/snort/figures and outputs/snort/tables")


if __name__ == "__main__":
    main()

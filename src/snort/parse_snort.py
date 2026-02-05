import re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "alerts_monday_to_friday"
OUT_DIR = ROOT / "outputs" / "snort"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# extended regex to include classification and priority
pattern = re.compile(
    r'^(?P<ts>\d{2}/\d{2}-\d{2}:\d{2}:\d{2}\.\d+).*?'
    r'\[(?P<gid>\d+):(?P<sid>\d+):(?P<rev>\d+)\]\s+'
    r'(?P<msg>.*?)\s+'
    r'\[Classification:\s*(?P<class>.*?)\]\s+'
    r'\[Priority:\s*(?P<priority>\d+)\]'
)

rows = []

for log_file in DATA_DIR.glob("alert_*.log"):
    day = log_file.stem.replace("alert_", "")
    with log_file.open(errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            rows.append(
                {
                    "day": day,
                    "timestamp": m.group("ts"),
                    "sid": int(m.group("sid")),
                    "priority": int(m.group("priority")),
                    "classification": m.group("class"),
                    "message": m.group("msg"),
                }
            )

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "snort_alerts.csv", index=False)

print("Parsed alerts:", len(df))

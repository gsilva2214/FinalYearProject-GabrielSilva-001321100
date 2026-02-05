from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# Project root = final-year-project
ROOT = Path(__file__).resolve().parents[1]

PIPELINE = [
    ("Snort parse", ROOT / "src" / "snort" / "parse_snort.py"),
    ("Anomaly features", ROOT / "src" / "anomaly" / "anomaly_features.py"),
    ("Anomaly train and score", ROOT / "src" / "anomaly" / "anomaly_train_score.py"),
    ("Anomaly vs Snort comparison", ROOT / "src" / "anomaly" / "anomaly_compare_snort.py"),
]


def run_step(name: str, script: Path) -> None:
    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")

    print("\n==============================")
    print("Running:", name)
    print("Script:", script)
    print("==============================")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
    )

    if result.returncode != 0:
        raise RuntimeError(f"Pipeline failed at step: {name}")


def main() -> None:
    print("Project root:", ROOT)

    for name, script in PIPELINE:
        run_step(name, script)

    print("\nPipeline finished successfully")
    print("Check outputs/ for tables and figures")


if __name__ == "__main__":
    main()

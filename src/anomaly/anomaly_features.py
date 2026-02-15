"""
Anomaly Feature Engineering - CIC-IDS 2017 CSV
================================================
Loads raw network flow features directly from the
CIC-IDS 2017 dataset. Independent of Snort.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def clean_cicids(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the CIC-IDS 2017 CSV."""

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    drop_cols = []
    for col in ["Flow ID"]:
        if col in df.columns:
            drop_cols.append(col)

    # Create binary label BEFORE dropping
    if "Label" not in df.columns:
        raise KeyError(
            f"No 'Label' column found. Columns: {list(df.columns)}"
        )

    df["label_original"] = df["Label"].str.strip()
    df["label_original"] = df["label_original"].str.replace("\x96", "-")
    df["label_binary"] = df["label_original"].apply(
        lambda x: 0 if x == "BENIGN" else 1
    )

    # Drop non-feature columns
    drop_cols.append("Label")
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Replace inf with NaN then drop
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return only numeric feature columns."""
    exclude = {"label_binary", "label_original"}
    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]
    return feature_cols


def run_anomaly_pipeline(
    input_path: str,
    output_csv: str,
    figures_dir: str,
    tables_dir: str,
    params: dict,
) -> str:
    """
    Load CIC-IDS CSV, clean it, save processed features.
    This replaces the old Snort-based feature pipeline.
    """

    Path(tables_dir).mkdir(parents=True, exist_ok=True)
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------
    # Load dataset
    # ------------------------------------------
    print(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Raw rows: {len(df)}")

    # ------------------------------------------
    # Clean
    # ------------------------------------------
    df = clean_cicids(df)
    print(f"Clean rows: {len(df)}")

    feature_cols = get_feature_columns(df)
    print(f"Feature columns: {len(feature_cols)}")

    # ------------------------------------------
    # Save dataset summary to tables
    # ------------------------------------------
    # Label distribution
    label_dist = df["label_original"].value_counts().reset_index()
    label_dist.columns = ["attack_type", "count"]
    label_dist.to_csv(
        Path(tables_dir) / "label_distribution.csv", index=False
    )

    # Binary split
    binary_dist = df["label_binary"].value_counts().reset_index()
    binary_dist.columns = ["label", "count"]
    binary_dist["label"] = binary_dist["label"].map(
        {0: "BENIGN", 1: "ATTACK"}
    )
    binary_dist.to_csv(
        Path(tables_dir) / "binary_distribution.csv", index=False
    )

    # Feature stats
    df[feature_cols].describe().to_csv(
        Path(tables_dir) / "feature_statistics.csv"
    )

    # ------------------------------------------
    # Save cleaned data for model training
    # ------------------------------------------
    df.to_csv(output_csv, index=False)
    print(f"Wrote cleaned features to: {output_csv}")

    return output_csv


def main() -> None:
    """Standalone runner for testing."""
    ROOT = Path(__file__).resolve().parents[2]
    INPUT = ROOT / "data" / "cicids_dataset.csv"
    OUTPUT = ROOT / "outputs" / "anomaly"

    run_anomaly_pipeline(
        input_path=str(INPUT),
        output_csv=str(OUTPUT / "anomaly_results.csv"),
        figures_dir=str(OUTPUT / "figures"),
        tables_dir=str(OUTPUT / "tables"),
        params={},
    )


if __name__ == "__main__":
    main()
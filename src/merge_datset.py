from pathlib import Path
import pandas as pd

# debug to help find the right folder
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "cicids_raw"
OUTPUT = ROOT / "data" / "cicids_dataset.csv"

print(f"Script location: {Path(__file__).resolve()}")
print(f"ROOT:            {ROOT}")
print(f"DATA_DIR:        {DATA_DIR}")
print(f"DATA_DIR exists: {DATA_DIR.exists()}")
print()

if DATA_DIR.exists():
    print("Files found in DATA_DIR:")
    for f in sorted(DATA_DIR.iterdir()):
        print(f"  {f.name}  ({f.suffix})")
    print()
else:
    print("DATA_DIR does not exist!")
    print("Check that this folder exists:")
    print(f"  {DATA_DIR}")
    exit(1)

files = sorted(DATA_DIR.glob("*pcap_ISCX*"))

if not files:
    files = sorted(DATA_DIR.glob("*"))
    files = [f for f in files if f.is_file()]

if not files:
    print("No files found at all.")
    exit(1)

print(f"Loading {len(files)} files:\n")

dfs = []
for f in files:
    print(f"Loading {f.name}...")
    df = pd.read_csv(f, low_memory=False, encoding="latin-1")
    df.columns = df.columns.str.strip()

    print(f"  Rows:    {len(df)}")
    print(f"  Columns: {len(df.columns)}")

    required = ["Source IP", "Destination IP", "Label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  WARNING MISSING: {missing}")
        print(f"  First 5 columns: {list(df.columns[:5])}")
        continue

    labels = df["Label"].str.strip().value_counts()
    print(f"  Labels:")
    for label, count in labels.items():
        print(f"    {label}: {count}")

    dfs.append(df)
    print()

if not dfs:
    print("No valid files loaded.")
    exit(1)

combined = pd.concat(dfs, ignore_index=True)
combined.columns = combined.columns.str.strip()

print(f"\n{'='*50}")
print(f"COMBINED DATASET")
print(f"{'='*50}")
print(f"Total rows:    {len(combined)}")
print(f"Total columns: {len(combined.columns)}")
print(f"\nLabel distribution:")
print(combined["Label"].str.strip().value_counts().to_string())
print(f"\nUnique Source IPs:      {combined['Source IP'].nunique()}")
print(f"Unique Destination IPs: {combined['Destination IP'].nunique()}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(OUTPUT, index=False)
print(f"\nSaved to: {OUTPUT}")
print(f"File size: {OUTPUT.stat().st_size / 1e6:.1f} MB")
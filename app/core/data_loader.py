import pandas as pd
import numpy as np
import os
from core import PATHS

def _read(path, sample_size=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    df = pd.read_csv(path, nrows=sample_size)
    df.columns = df.columns.str.strip()
    return df

def load_anomaly_results(sample_size=None):
    return _read(PATHS["anomaly_results"], sample_size)

def load_snort_alerts():
    return _read(PATHS["snort_alerts"])
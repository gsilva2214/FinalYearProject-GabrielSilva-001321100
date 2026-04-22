"""
core modules for the dashboard.
"""
import os

THIS_FILE = os.path.abspath(__file__)
CORE_DIR = os.path.dirname(THIS_FILE)
APP_DIR = os.path.dirname(CORE_DIR)
PROJECT_ROOT = os.path.dirname(APP_DIR)

PATHS = {
    "dataset":          os.path.join(PROJECT_ROOT, "data", "cicids_dataset.csv"),
    "dataset_dir":      os.path.join(PROJECT_ROOT, "data", "cicids_raw"),
    "snort_alerts":     os.path.join(PROJECT_ROOT, "data", "snort_alerts.csv"),
    "snort_detailed":   os.path.join(PROJECT_ROOT, "outputs", "snort", "snort_alerts.csv"),
    "anomaly_results":  os.path.join(PROJECT_ROOT, "outputs", "anomaly", "anomaly_results.csv"),
    "compare_dir":      os.path.join(PROJECT_ROOT, "outputs", "compare"),
    "snort_output_dir": os.path.join(PROJECT_ROOT, "outputs", "snort"),
    "anomaly_figures":  os.path.join(PROJECT_ROOT, "outputs", "anomaly", "figures"),
    "anomaly_tables":   os.path.join(PROJECT_ROOT, "outputs", "anomaly", "tables"),
    "docs":             os.path.join(PROJECT_ROOT, "docs"),
    # Live detector model files
    "models_dir":       os.path.join(PROJECT_ROOT, "models"),
    "model_file":       os.path.join(PROJECT_ROOT, "models", "isolation_forest.joblib"),
    "scaler_file":      os.path.join(PROJECT_ROOT, "models", "scaler.joblib"),
    "feature_meta":     os.path.join(PROJECT_ROOT, "models", "feature_meta.json"),
}
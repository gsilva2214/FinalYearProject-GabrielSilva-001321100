import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Home", layout="wide", initial_sidebar_state="expanded")

st.title("About This Project")

st.markdown("""
### What is this?
This program compares two different network intrusion detection approaches. Those being a rule-based system (Snort)
and an anomaly-based model (Isolation Forest).

### Pages
**The main page - Live Detector**
Upload a CSV file in CIC-IDS 2017 format and both engines will classify every flow in real time.
This is the only page that runs live — everything else displays pre-computed results.

**Head to Head Results**
Compares Snort and the Isolation Forest side by side using precision, recall, F1, and false positive rate.
Results were pre-computed on 2.8 million real network flows.

**Hybrid Fusion Results**
Explores four strategies for combining both detection engines.
Toggle between strategies and see how the metrics and radar chart change.
""")


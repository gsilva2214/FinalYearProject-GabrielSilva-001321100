import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT         = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANOMALY_TABS = os.path.join(ROOT, "outputs", "anomaly", "tables")
SNORT_TABS   = os.path.join(ROOT, "outputs", "snort",   "tables")

st.title("Head-to-Head Comparison")
st.markdown("**Snort (Rule-Based)** vs **Isolation Forest (Anomaly-Based)**")

def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None

def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None

anomaly_m = load_json(os.path.join(ANOMALY_TABS, "anomaly_metrics.json"))
snort_m   = load_json(os.path.join(SNORT_TABS,   "snort_metrics.json"))

if not anomaly_m or not snort_m:
    st.error("Run `anomaly_train_score.py` and `snort_evaluate.py` first.")
    st.stop()

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Snort — Rule-Based")
    a, b, c, d = st.columns(4)
    a.metric("F1",        f"{snort_m['f1_score']:.3f}")
    b.metric("Precision", f"{snort_m['precision']:.3f}")
    c.metric("Recall",    f"{snort_m['recall']:.3f}")
    d.metric("FPR",       f"{snort_m['false_positive_rate']:.3f}")
with col2:
    st.markdown("### Isolation Forest — Anomaly-Based")
    a, b, c, d = st.columns(4)
    a.metric("F1",        f"{anomaly_m['f1_score']:.3f}")
    b.metric("Precision", f"{anomaly_m['precision']:.3f}")
    c.metric("Recall",    f"{anomaly_m['recall']:.3f}")
    d.metric("FPR",       f"{anomaly_m['false_positive_rate']:.3f}")

st.markdown("---")
st.subheader("Metrics — Side by Side")
keys   = ["precision", "recall", "f1_score", "false_positive_rate", "detection_rate"]
labels = ["Precision", "Recall", "F1 Score", "False Positive Rate", "Detection Rate"]
fig = go.Figure()
fig.add_trace(go.Bar(name="Snort",             x=labels, y=[float(snort_m.get(k, 0))   for k in keys], marker_color="#2196F3"))
fig.add_trace(go.Bar(name="Isolation Forest",  x=labels, y=[float(anomaly_m.get(k, 0)) for k in keys], marker_color="#FF9800"))
fig.update_layout(barmode="group", yaxis=dict(range=[0, 1.1]), legend=dict(orientation="h", y=1.1), margin=dict(t=10))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Confusion Matrices")
col1, col2 = st.columns(2)

def cm_fig(tp, fp, tn, fn, title, colour):
    f = go.Figure(go.Heatmap(z=np.array([[tn, fp], [fn, tp]]),
        x=["Predicted BENIGN", "Predicted ATTACK"], y=["Actual BENIGN", "Actual ATTACK"],
        text=[[f"{tn:,}", f"{fp:,}"], [f"{fn:,}", f"{tp:,}"]], texttemplate="%{text}",
        colorscale=colour, showscale=False))
    f.update_layout(title=title, height=320, margin=dict(t=40, b=10))
    return f

with col1:
    st.plotly_chart(cm_fig(snort_m["true_positives"], snort_m["false_positives"], snort_m["true_negatives"], snort_m["false_negatives"], "Snort", "Blues"), use_container_width=True)
with col2:
    st.plotly_chart(cm_fig(anomaly_m["true_positives"], anomaly_m["false_positives"], anomaly_m["true_negatives"], anomaly_m["false_negatives"], "Isolation Forest", "Oranges"), use_container_width=True)

st.markdown("---")
st.subheader("Detection Rate by Attack Type")
anom_att  = load_csv(os.path.join(ANOMALY_TABS, "per_attack_detection.csv"))
snort_att = load_csv(os.path.join(SNORT_TABS,   "snort_per_attack_detection.csv"))

if anom_att is not None and snort_att is not None:
    merged = anom_att.rename(columns={"detection_rate": "ML %"}).merge(snort_att.rename(columns={"detection_rate": "Snort %"}), on="attack_type", how="outer")
    merged = merged[merged["attack_type"] != "BENIGN"].fillna(0)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Snort", x=merged["attack_type"], y=merged["Snort %"], marker_color="#2196F3"))
    fig2.add_trace(go.Bar(name="ML",    x=merged["attack_type"], y=merged["ML %"],    marker_color="#FF9800"))
    fig2.update_layout(barmode="group", xaxis_tickangle=-35, yaxis_title="Detection Rate (%)", legend=dict(orientation="h", y=1.1), margin=dict(t=10))
    st.plotly_chart(fig2, use_container_width=True)
    merged["Winner"] = merged.apply(lambda r: "Snort" if r["Snort %"] > r["ML %"] else ("ML" if r["ML %"] > r["Snort %"] else "—"), axis=1)
    st.dataframe(merged[["attack_type", "Snort %", "ML %", "Winner"]].reset_index(drop=True), use_container_width=True, hide_index=True)

st.markdown("---")
st.info("**Neither system is sufficient alone.**")
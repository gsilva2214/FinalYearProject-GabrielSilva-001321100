import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.fusion import STRATEGIES, DESCRIPTIONS, run_fusion
from core.metrics import calculate_all_metrics
from sklearn.metrics import confusion_matrix as sk_cm

ROOT          = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANOMALY_PREDS = os.path.join(ROOT, "outputs", "anomaly", "tables", "anomaly_predictions.csv")
SNORT_PREDS   = os.path.join(ROOT, "outputs", "snort",   "tables", "snort_predictions.csv")

st.title("Hybrid Fusion Analysis")
st.markdown("Combining both engines using four strategies.")

if not os.path.exists(ANOMALY_PREDS) or not os.path.exists(SNORT_PREDS):
    st.error("Run `anomaly_train_score.py` and `snort_evaluate.py` first.")
    st.stop()

@st.cache_data
def load_preds():
    a = pd.read_csv(ANOMALY_PREDS)
    s = pd.read_csv(SNORT_PREDS)
    n = min(len(a), len(s))
    return a.iloc[:n].reset_index(drop=True), s.iloc[:n].reset_index(drop=True)

anom_df, snort_df = load_preds()
y_true  = anom_df["label_binary"].values
y_ml    = anom_df["ml_prediction"].values
y_snort = snort_df["snort_prediction"].values

st.success(f"Loaded {len(anom_df):,} flow predictions.")
st.markdown("---")

selected = st.radio("Strategy", STRATEGIES)
st.markdown(DESCRIPTIONS[selected])

a_weight, r_weight, threshold = 0.6, 0.4, 0.5
if selected == "Weighted Voting":
    c1, c2, c3 = st.columns(3)
    a_weight  = c1.slider("ML Weight",         0.0, 1.0, 0.6, 0.05)
    r_weight  = c2.slider("Snort Weight",       0.0, 1.0, 0.4, 0.05)
    threshold = c3.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)

y_hybrid = run_fusion(y_ml, y_snort, selected, a_weight, r_weight, threshold)
m = calculate_all_metrics(y_true, y_hybrid)

st.markdown("---")
st.subheader(f"Results — {selected}")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Precision", f"{m['precision']:.3f}")
c2.metric("Recall",    f"{m['recall']:.3f}")
c3.metric("F1 Score",  f"{m['f1']:.3f}")
c4.metric("FPR",       f"{m['fpr']:.3f}")
c5.metric("Accuracy",  f"{m['accuracy']:.3f}")

tn, fp, fn, tp = sk_cm(y_true, y_hybrid).ravel()
fig_cm = go.Figure(go.Heatmap(z=np.array([[tn, fp], [fn, tp]]),
    x=["Predicted BENIGN", "Predicted ATTACK"], y=["Actual BENIGN", "Actual ATTACK"],
    text=[[f"{tn:,}", f"{fp:,}"], [f"{fn:,}", f"{tp:,}"]], texttemplate="%{text}",
    colorscale="Purples", showscale=False))
fig_cm.update_layout(title=f"Confusion Matrix — {selected}", height=320, margin=dict(t=40))
st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("---")
st.subheader("All Strategies — Side by Side")

rows = []
for strat in STRATEGIES + ["Snort only", "ML only"]:
    preds = run_fusion(y_ml, y_snort, strat) if strat in STRATEGIES else (y_snort if strat == "Snort only" else y_ml)
    ms = calculate_all_metrics(y_true, preds)
    _tn, _fp, _fn, _tp = sk_cm(y_true, preds).ravel()
    rows.append({"Strategy": strat, "Precision": round(ms["precision"], 4), "Recall": round(ms["recall"], 4),
                 "F1 Score": round(ms["f1"], 4), "FPR": round(ms["fpr"], 4), "Accuracy": round(ms["accuracy"], 4),
                 "TP": int(_tp), "FP": int(_fp), "TN": int(_tn), "FN": int(_fn)})

all_df = pd.DataFrame(rows)
st.dataframe(all_df.style.apply(lambda r: ["background-color: #fff3e0"] * len(r) if r["Strategy"] == selected else [""] * len(r), axis=1),
             use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("Radar Chart")
radar_df = all_df.copy()
radar_df["1 - FPR"] = 1 - radar_df["FPR"]
cats = ["Precision", "Recall", "F1 Score", "Accuracy", "1 - FPR"]
colours = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50", "#F44336", "#009688"]
fig_r = go.Figure()
for i, row in radar_df.iterrows():
    vals = [row[c] for c in cats] + [row[cats[0]]]
    fig_r.add_trace(go.Scatterpolar(r=vals, theta=cats + [cats[0]], name=row["Strategy"], line_color=colours[i % len(colours)]))
fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=460, margin=dict(t=20))
st.plotly_chart(fig_r, use_container_width=True)

best = all_df.loc[all_df["F1 Score"].idxmax()]
st.info(f"**Best F1:** {best['Strategy']} ({best['F1 Score']:.3f}) — The Tiered strategy best mirrors a real SOC workflow, reducing alert fatigue while maintaining coverage.")
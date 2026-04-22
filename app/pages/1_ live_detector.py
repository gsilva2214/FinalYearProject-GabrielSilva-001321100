import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.fusion import run_fusion, STRATEGIES

ROOT        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR   = os.path.join(ROOT, "models")

st.title("Live Network Flow Detector")
st.markdown("Upload a CSV — both engines classify every flow instantly.")

@st.cache_resource
def load_model():
    if not os.path.exists(os.path.join(MODEL_DIR, "isolation_forest.joblib")):
        return None, None, None
    return (joblib.load(os.path.join(MODEL_DIR, "isolation_forest.joblib")),
            joblib.load(os.path.join(MODEL_DIR, "scaler.joblib")),
            json.load(open(os.path.join(MODEL_DIR, "feature_meta.json"))))

model, scaler, meta = load_model()
if model is None:
    st.error("No model found. Run `anomaly_train_score.py` first.")
    st.stop()

st.success(f"Model loaded — Isolation Forest ({meta['n_estimators']} estimators, {meta['n_features']} features)")

class RuleEngine:
    PORTS = {22, 23, 80, 443, 3389, 21, 25, 53, 8080, 8443, 445, 1433, 3306}
    def _c(self, row, *names):
        for n in names:
            if n in row.index:
                try: return float(row[n])
                except: return 0.0
        return 0.0
    def check(self, row):
        fired, c = [], self._c
        dst  = int(c(row, "Destination Port", "Dst Port"))
        fp   = c(row, "Total Fwd Packets")
        bp   = c(row, "Total Backward Packets")
        pl   = c(row, "Average Packet Size")
        dur  = c(row, "Flow Duration")
        bps  = c(row, "Flow Bytes/s")
        pps  = c(row, "Flow Packets/s")
        syn  = c(row, "SYN Flag Count")
        rst  = c(row, "RST Flag Count")
        fin  = c(row, "FIN Flag Count")
        urg  = c(row, "URG Flag Count")
        ims  = c(row, "Flow IAT Mean")
        std  = c(row, "Flow IAT Std")
        win  = c(row, "Init_Win_bytes_forward", "Init Fwd Win Byts")
        if pps > 100000:                                        fired.append("SID:1001 DoS — packet rate")
        if bps > 5_000_000:                                     fired.append("SID:1002 DoS — byte rate")
        if syn > 10 and pl < 100 and dur < 1_000_000:          fired.append("SID:1003 SYN Flood")
        if rst > 5 and fp < 3:                                  fired.append("SID:1004 Port scan")
        if dst in self.PORTS and fp > 50:                       fired.append(f"SID:1005 Brute force :{dst}")
        if dst in {80,443,8080} and dur>10_000_000 and pl<200:  fired.append("SID:1006 Slowloris")
        if urg > 2:                                             fired.append("SID:1007 URG abuse")
        if 0 < std < 100 and ims > 0 and fp > 20:              fired.append("SID:1008 Botnet beacon")
        if fp == 0 and bp == 0:                                 fired.append("SID:1009 Zero-packet probe")
        if fin > 5 and dur < 500_000:                           fired.append("SID:1010 FIN flood")
        if 0 < win < 8:                                         fired.append("SID:1011 Tiny TCP window")
        if fp > 100 and bp == 0:                                fired.append("SID:1012 Asymmetric flood")
        return (1 if fired else 0), fired

engine = RuleEngine()

strategy = st.sidebar.radio("Fusion Strategy", STRATEGIES)
a_weight, r_weight, threshold = 0.6, 0.4, 0.5
if strategy == "Weighted Voting":
    a_weight  = st.sidebar.slider("ML Weight",         0.0, 1.0, 0.6, 0.05)
    r_weight  = st.sidebar.slider("Snort Weight",       0.0, 1.0, 0.4, 0.05)
    threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)

st.markdown("---")
uploaded = st.file_uploader("Upload CIC-IDS 2017 format CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

@st.cache_data
def load_csv(b):
    import io
    df = pd.read_csv(io.BytesIO(b), low_memory=False)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

raw_df = load_csv(uploaded.read())
st.success(f"Loaded **{len(raw_df):,} flows** from `{uploaded.name}`")

with st.spinner("Running both engines..."):
    t = time.time()
    X = np.nan_to_num(raw_df.reindex(columns=meta["feature_columns"], fill_value=0.0).values.astype(float))
    ml_preds  = np.where(model.predict(scaler.transform(X)) == -1, 1, 0)
    ml_scores = -model.decision_function(scaler.transform(X))
    snort_preds, snort_rules = zip(*[engine.check(row) for _, row in raw_df.iterrows()])
    snort_preds = np.array(snort_preds)
    snort_rules = ["; ".join(r) if r else "—" for r in snort_rules]
    hybrid = run_fusion(ml_preds, snort_preds, strategy, a_weight, r_weight, threshold)
    elapsed = time.time() - t

st.markdown("---")
total, n_att = len(hybrid), int(hybrid.sum())
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Flows",   f"{total:,}")
c2.metric("ATTACK",     f"{n_att:,}",          delta=f"{n_att/total*100:.1f}%", delta_color="inverse")
c3.metric("BENIGN",     f"{total-n_att:,}",    delta=f"{(total-n_att)/total*100:.1f}%")
c4.metric("ML Flagged",    f"{int(ml_preds.sum()):,}")
c5.metric("Snort Flagged", f"{int(snort_preds.sum()):,}")
st.caption(f"⚡ {elapsed:.2f}s | Strategy: **{strategy}**")

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    agree_df = pd.DataFrame({"Scenario": ["Both","ML only","Snort only","Neither XX"],
        "Count": [int(((ml_preds==1)&(snort_preds==1)).sum()), int(((ml_preds==1)&(snort_preds==0)).sum()),
                  int(((ml_preds==0)&(snort_preds==1)).sum()), int(((ml_preds==0)&(snort_preds==0)).sum())]})
    st.plotly_chart(px.bar(agree_df, x="Scenario", y="Count", color="Scenario", text="Count",
        color_discrete_sequence=["#4CAF50","#FF9800","#2196F3","#9E9E9E"]).update_layout(showlegend=False, margin=dict(t=10)),
        use_container_width=True)
with col2:
    fig_s = go.Figure()
    fig_s.add_trace(go.Histogram(x=ml_scores[hybrid==0], name="BENIGN", opacity=0.6, nbinsx=50, marker_color="#4CAF50"))
    fig_s.add_trace(go.Histogram(x=ml_scores[hybrid==1], name="ATTACK", opacity=0.6, nbinsx=50, marker_color="#F44336"))
    fig_s.update_layout(barmode="overlay", xaxis_title="Anomaly Score", margin=dict(t=10))
    st.plotly_chart(fig_s, use_container_width=True)

st.markdown("---")
results = raw_df.copy()
results["Verdict"] = ["ATTACK" if h else "BENIGN" for h in hybrid]
results["ML_Score"] = np.round(ml_scores, 4)
results["Rules_Fired"] = snort_rules

c1, c2 = st.columns(2)
show_attacks = c1.checkbox("Attacks only")
show_benign  = c2.checkbox("Benign only")
display = results[hybrid == 1] if (show_attacks and not show_benign) else results[hybrid == 0] if (show_benign and not show_attacks) else results
cols = ["Verdict", "ML_Score", "Rules_Fired"] + [c for c in ["Destination Port", "label_original", "Label"] if c in results.columns]
st.dataframe(display[cols].head(500), use_container_width=True, hide_index=True)

st.download_button("Download Results CSV", results.to_csv(index=False).encode(), f"results_{uploaded.name}", "text/csv")
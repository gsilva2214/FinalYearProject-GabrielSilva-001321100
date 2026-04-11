import streamlit as st
import pandas as pd
import plotly.express as px
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.data_loader import load_dataset, get_dataset_summary

st.title("Dataset Overview")
st.markdown("**CIC-IDS 2017** — 2.8 million real network flows, 14 attack categories")

sample_size = st.sidebar.slider("Sample size", 5000, 200000, 50000, 5000)

@st.cache_data
def load(n):
    return load_dataset(sample_size=n)

df = load(sample_size)
summary = get_dataset_summary(df)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Flows",  "2,827,876")
c2.metric("Features",     "80")
c3.metric("Benign",       "2,271,320")
c4.metric("Attacks",      "556,556")
c5.metric("Attack Types", "14")

st.markdown("---")

if summary["class_distribution"]:
    dist = pd.DataFrame({"Type": list(summary["class_distribution"].keys()), "Count": list(summary["class_distribution"].values())}).sort_values("Count", ascending=False)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Flows per Category")
        fig = px.bar(dist, x="Type", y="Count", color="Type", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, xaxis_tickangle=-40, xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Benign vs Attack Split")
        fig2 = px.pie(pd.DataFrame({"Category": ["Benign", "Attack"], "Count": [summary["benign_count"], summary["attack_count"]]}),
                      names="Category", values="Count", hole=0.4,
                      color_discrete_map={"Benign": "#2196F3", "Attack": "#E53935"})
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig2, use_container_width=True)

with st.expander("View raw data sample"):
    st.dataframe(df.head(100), use_container_width=True)
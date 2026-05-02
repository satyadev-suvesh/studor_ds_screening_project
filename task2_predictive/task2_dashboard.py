"""
PathAI Staff Alert Dashboard (Streamlit)
========================================
Run: streamlit run task2_dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="PathAI Early Warning", layout="wide")

st.title("PathAI Early Warning Dashboard")
st.markdown("Weekly disengagement risk alerts for university advisors")

# Load data
output_dir = Path(__file__).resolve().parent.parent / "output" / "task2_output"
try:
    scores = pd.read_csv(output_dir / "engagement_scores_week6.csv")
    alerts = pd.read_csv(output_dir / "sample_alerts.csv")
except FileNotFoundError:
    st.error("Run task2_model.py first to generate the required data.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
tier_filter = st.sidebar.multiselect(
    "Risk Tier", ["Red", "Amber", "Green"], default=["Red", "Amber"])
course_filter = st.sidebar.selectbox(
    "Course", ["All"] + sorted(scores["code_module"].unique().tolist())
    if "code_module" in scores.columns else ["All"])

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
if "risk_tier" in scores.columns:
    col1.metric("Total Students", len(scores))
    col2.metric("Red Alerts",
                len(scores[scores.risk_tier == "Red"]),
                delta=None)
    col3.metric("Amber Alerts",
                len(scores[scores.risk_tier == "Amber"]))
    col4.metric("Green (Safe)",
                len(scores[scores.risk_tier == "Green"]))

# Alert table
st.subheader("Student Risk Alerts")
if len(alerts) > 0:
    filtered = alerts[alerts["Tier"].isin(tier_filter)]
    st.dataframe(filtered, use_container_width=True)

# Student lookup
st.subheader("Individual Student Lookup")
if "id_student" in scores.columns:
    sid = st.selectbox("Select Student ID",
                       sorted(scores["id_student"].unique()))
    stu = scores[scores.id_student == sid]
    if len(stu) > 0:
        row = stu.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Score",
                  f"{row.get('risk_prob', 0)*100:.1f}%")
        c2.metric("Risk Tier", row.get("risk_tier", "N/A"))
        c3.metric("Top Risk Factor",
                  row.get("top_risk_driver", "N/A"))

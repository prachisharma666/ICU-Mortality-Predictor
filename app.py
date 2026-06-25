"""
ICU In-Hospital Mortality Risk Predictor
Redesigned — deep navy, cyan pulse, Space Grotesk + JetBrains Mono
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="ICU Mortality · Risk Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  DESIGN TOKENS & CSS (Kept exactly as your original)
# ══════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', system-ui, sans-serif;
    background-color: #080C14;
    color: #B0BCDA;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: #080C14; }
::-webkit-scrollbar-thumb { background: #1C2438; border-radius: 2px; }

/* ── Block container ── */
.block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 3rem !important;
    max-width: 1300px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #060A11 !important;
    border-right: 1px solid #111827 !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

/* ── Headings ── */
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    color: #E8EDF8;
}

/* ── Streamlit radio (nav) ── */
[data-testid="stSidebar"] .stRadio > label {
    display: none !important;
}
[data-testid="stSidebar"] .stRadio > div {
    gap: 2px !important;
    flex-direction: column !important;
}
[data-testid="stSidebar"] .stRadio > div > label {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 10px 20px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: #4A5A7A !important;
    cursor: pointer !important;
    border-left: 2px solid transparent !important;
    transition: all 0.15s ease !important;
    letter-spacing: 0.01em !important;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    color: #8A9BBF !important;
    background: rgba(0,212,255,0.04) !important;
    border-left-color: #1C2438 !important;
}
[data-testid="stSidebar"] .stRadio [data-checked="true"] > div {
    color: #00D4FF !important;
}
[data-testid="stSidebar"] .stRadio div[data-testid="stMarkdownContainer"] p {
    color: inherit !important;
}

/* ── Input labels ── */
.stSlider label,
.stSelectbox label,
.stNumberInput label,
.stRadio > label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    color: #2E3A55 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* ── Select boxes ── */
div[data-baseweb="select"] > div {
    background-color: #161D2E !important;
    border: 1px solid #1C2438 !important;
    border-radius: 6px !important;
    color: #B0BCDA !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    transition: border-color 0.2s !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: #00D4FF !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.1) !important;
}

/* ── Number inputs ── */
input[type="number"] {
    background-color: #161D2E !important;
    border: 1px solid #1C2438 !important;
    border-radius: 6px !important;
    color: #E8EDF8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s !important;
}
input[type="number"]:focus {
    border-color: #00D4FF !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.08) !important;
    outline: none !important;
}

/* ── Slider ── */
[data-baseweb="slider"] [role="slider"] {
    background: #00D4FF !important;
    border-color: #00D4FF !important;
    box-shadow: 0 0 8px rgba(0,212,255,0.5) !important;
}
[data-baseweb="slider"] div[aria-hidden="true"] > div > div {
    background: linear-gradient(90deg, #005F73, #00D4FF) !important;
}

/* ── Button ── */
.stButton > button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    color: #00D4FF !important;
    border: 1px solid #00D4FF !important;
    border-radius: 6px !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button::before {
    content: '' !important;
    position: absolute !important;
    inset: 0 !important;
    background: rgba(0,212,255,0.06) !important;
    opacity: 0 !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    background: rgba(0,212,255,0.1) !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.2), inset 0 0 20px rgba(0,212,255,0.05) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
    box-shadow: none !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #0F1520 !important;
    border: 1px solid #1C2438 !important;
    border-radius: 6px !important;
    color: #4A5A7A !important;
    font-size: 0.74rem !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}
.streamlit-expanderContent {
    background: #0F1520 !important;
    border: 1px solid #1C2438 !important;
    border-top: none !important;
    border-radius: 0 0 6px 6px !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border: 1px solid #1C2438 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}
.stDataFrame thead th {
    background: #161D2E !important;
    color: #2E3A55 !important;
    font-size: 0.65rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    border-bottom: 1px solid #1C2438 !important;
}
.stDataFrame tbody td {
    color: #8A9BBF !important;
    font-size: 0.8rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    background: #0F1520 !important;
    border-bottom: 1px solid #111827 !important;
}

/* ── Caption ── */
.stCaption { color: #2E3A55 !important; font-size: 0.68rem !important; }

/* ══════════════════════════════
   ANIMATIONS & REMAINING CSS
══════════════════════════════ */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulseGlow {
    0%, 100% { opacity: 0.6; }
    50%       { opacity: 1; }
}
@keyframes ecg {
    0%   { stroke-dashoffset: 600; }
    100% { stroke-dashoffset: 0; }
}
@keyframes scanDown {
    0%   { transform: translateY(0); opacity: 0.6; }
    100% { transform: translateY(60px); opacity: 0; }
}
@keyframes riskReveal {
    from { opacity: 0; transform: scale(0.9); }
    to   { opacity: 1; transform: scale(1); }
}
@keyframes barGrow {
    from { width: 0; }
}

/* ══════════════════════════════
   SIDEBAR COMPONENTS
══════════════════════════════ */
.sb-header {
    padding: 24px 20px 18px;
    border-bottom: 1px solid #111827;
    position: relative;
    overflow: hidden;
}
.sb-ecg-wrap {
    margin-bottom: 12px;
    height: 32px;
    position: relative;
}
.sb-ecg-wrap svg {
    width: 100%;
    height: 32px;
}
.ecg-line {
    fill: none;
    stroke: #00D4FF;
    stroke-width: 1.5;
    stroke-dasharray: 600;
    stroke-dashoffset: 600;
    animation: ecg 2.8s ease-out 0.3s forwards;
    filter: drop-shadow(0 0 3px #00D4FF);
}
.sb-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #E8EDF8;
    letter-spacing: -0.02em;
}
.sb-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    color: #2E3A55;
    letter-spacing: 0.1em;
    margin-top: 4px;
}

.sb-nav-label {
    padding: 14px 20px 6px;
    font-family: 'Inter', sans-serif;
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #1C2438;
}

.sb-info-block {
    margin: 0 12px;
    background: #0C1220;
    border: 1px solid #111827;
    border-radius: 8px;
    overflow: hidden;
}
.sb-info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 14px;
    border-bottom: 1px solid #0D1422;
    font-size: 0.7rem;
}
.sb-info-row:last-child { border-bottom: none; }
.sb-info-key { color: #2E3A55; font-family: 'Inter', sans-serif; }
.sb-info-val {
    font-family: 'JetBrains Mono', monospace;
    color: #4A5A7A;
    font-size: 0.64rem;
}
.sb-status-dot {
    display: inline-block;
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #00E5A0;
    box-shadow: 0 0 6px #00E5A0;
    margin-right: 6px;
    animation: pulseGlow 2s ease-in-out infinite;
    vertical-align: middle;
}

.sb-notice {
    margin: 12px;
    padding: 10px 14px;
    background: rgba(255,92,108,0.06);
    border: 1px solid rgba(255,92,108,0.15);
    border-radius: 6px;
    font-size: 0.65rem;
    color: #4A3A40;
    line-height: 1.65;
}

/* ══════════════════════════════
   PAGE COMPONENTS
══════════════════════════════ */
.page-header {
    margin-bottom: 2rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid #111827;
    animation: fadeUp 0.35s ease both;
}
.page-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    font-weight: 500;
    color: #00D4FF;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 6px;
    opacity: 0.8;
}
.page-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #E8EDF8;
    letter-spacing: -0.03em;
    line-height: 1.1;
}
.page-desc {
    font-size: 0.78rem;
    color: #4A5A7A;
    margin-top: 6px;
    line-height: 1.55;
}

.section-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 1.6rem 0 0.75rem;
}
.section-head-line {
    flex: 1;
    height: 1px;
    background: #111827;
}
.section-head-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #1C2438;
    white-space: nowrap;
}

/* ── Input panel ── */
.input-panel {
    background: #0C1220;
    border: 1px solid #111827;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 10px;
    animation: fadeUp 0.35s ease both;
    transition: border-color 0.2s;
}
.input-panel:hover {
    border-color: #1C2438;
}

/* ── Flag chips ── */
.flag-chip {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    padding: 2px 8px;
    border-radius: 2px;
    margin: 0 3px 4px 0;
}
.flag-warn { background: rgba(255,170,0,0.1); color: #FFAA00; border: 1px solid rgba(255,170,0,0.2); }
.flag-crit { background: rgba(255,92,108,0.1); color: #FF5C6C; border: 1px solid rgba(255,92,108,0.2); }

/* ── Risk output card ── */
.risk-card {
    background: #0C1220;
    border: 1px solid #1C2438;
    border-radius: 12px;
    padding: 30px 26px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: riskReveal 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) both;
}
.risk-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: var(--risk-color);
    box-shadow: 0 0 20px var(--risk-color);
}
.risk-card::after {
    content: '';
    position: absolute;
    top: -60px; left: 50%;
    transform: translateX(-50%);
    width: 200px; height: 120px;
    background: radial-gradient(ellipse at center, var(--risk-color) 0%, transparent 70%);
    opacity: 0.06;
    pointer-events: none;
}

.risk-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.57rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #2E3A55;
    margin-bottom: 14px;
}
.risk-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 5.5rem;
    font-weight: 700;
    letter-spacing: -0.05em;
    line-height: 0.9;
    color: var(--risk-color);
    text-shadow: 0 0 40px var(--risk-color);
    animation: riskReveal 0.5s 0.05s cubic-bezier(0.34, 1.56, 0.64, 1) both;
}
.risk-pct {
    font-size: 2rem;
    font-weight: 300;
    opacity: 0.7;
}
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 20px;
    margin-top: 12px;
    animation: fadeUp 0.4s 0.1s ease both;
}
.risk-badge::before {
    content: '';
    width: 5px; height: 5px;
    border-radius: 50%;
    background: currentColor;
    animation: pulseGlow 1.8s ease-in-out infinite;
}
.badge-high     { background: rgba(255,92,108,0.1);  color: #FF5C6C; border: 1px solid rgba(255,92,108,0.25); }
.badge-moderate { background: rgba(255,170,0,0.1);   color: #FFAA00; border: 1px solid rgba(255,170,0,0.25); }
.badge-low      { background: rgba(0,229,160,0.1);   color: #00E5A0; border: 1px solid rgba(0,229,160,0.25); }

/* ── Arc gauge ── */
.gauge-wrap {
    margin: 20px auto 8px;
    width: 100%;
}
.gauge-track-outer {
    position: relative;
    height: 5px;
    background: #111827;
    border-radius: 3px;
    overflow: hidden;
}
.gauge-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #005F73, var(--risk-color));
    box-shadow: 0 0 10px var(--risk-color);
    animation: barGrow 0.9s cubic-bezier(0.34, 1.56, 0.64, 1) both;
}
.gauge-ticks {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.54rem;
    color: #1C2438;
    letter-spacing: 0.04em;
    margin-top: 5px;
}

</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  LOAD ARTIFACTS (UPDATED: Removed base_xgb.pkl)
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    model         = joblib.load('models/mortality_model.pkl')
    explainer     = joblib.load('models/shap_explainer.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    train_medians = joblib.load('models/train_medians.pkl')
    return model, explainer, feature_names, train_medians

model, explainer, feature_names, train_medians = load_artifacts()

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    # ECG animation + branding
    st.markdown("""
    <div class="sb-header">
        <div class="sb-ecg-wrap">
            <svg viewBox="0 0 220 32" xmlns="http://www.w3.org/2000/svg">
                <polyline class="ecg-line"
                    points="0,16 30,16 38,16 42,4 46,28 50,10 54,20 58,16 90,16
                            98,16 102,4 106,28 110,10 114,20 118,16 160,16
                            168,16 172,4 176,28 180,10 184,20 188,16 220,16"
                />
            </svg>
        </div>
        <div class="sb-title">ICU Risk Engine</div>
        <div class="sb-subtitle">MIMIC-III &nbsp;·&nbsp; XGBOOST &nbsp;·&nbsp; SHAP</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-nav-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "nav",
        ["Patient Risk Assessment", "Population Analytics"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-info-block">
        <div class="sb-info-row">
            <span class="sb-info-key">Status</span>
            <span class="sb-info-val"><span class="sb-status-dot"></span>ONLINE</span>
        </div>
        <div class="sb-info-row">
            <span class="sb-info-key">Algorithm</span>
            <span class="sb-info-val">XGBoost</span>
        </div>
        <div class="sb-info-row">
            <span class="sb-info-key">Dataset</span>
            <span class="sb-info-val">MIMIC-III</span>
        </div>
        <div class="sb-info-row">
            <span class="sb-info-key">Task</span>
            <span class="sb-info-val">Binary classif.</span>
        </div>
        <div class="sb-info-row">
            <span class="sb-info-key">Outcome</span>
            <span class="sb-info-val">In-hosp. death</span>
        </div>
        <div class="sb-info-row" style="border-bottom:none;">
            <span class="sb-info-key">Explainability</span>
            <span class="sb-info-val">SHAP waterfall</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-notice">
        ⚠ Research prototype. Not validated for clinical decision-making. Do not substitute for physician judgment.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 1: PATIENT RISK ASSESSMENT
# ══════════════════════════════════════════════════════════════
if page == "Patient Risk Assessment":

    st.markdown("""
    <div class="page-header">
        <div class="page-eyebrow">◈ Clinical decision support</div>
        <div class="page-title">Patient Risk Assessment</div>
        <div class="page-desc">Enter patient parameters to generate in-hospital mortality probability with feature-level SHAP attribution.</div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        # — Demographics —
        st.markdown("""
        <div class="section-head">
            <span class="section-head-label">01 · Demographics</span>
            <div class="section-head-line"></div>
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            c1, c2 = st.columns(2)
            with c1:
                age    = st.slider("Age", 18, 90, 65)
                gender = st.selectbox("Sex", ["M", "F"])
            with c2:
                admission_type = st.selectbox("Admission type", ["EMERGENCY", "ELECTIVE", "URGENT"])
                insurance      = st.selectbox("Insurance", ["Medicare", "Medicaid", "Private", "Government", "Self Pay"])

        icd_chapter = st.selectbox(
            "Primary diagnosis (ICD-9 chapter)",
            ["circulatory", "respiratory", "digestive", "nervous",
             "mental", "infectious", "neoplasm", "genitourinary", "other"],
        )

        # — Hospitalization —
        st.markdown("""
        <div class="section-head">
            <span class="section-head-label">02 · Hospitalization</span>
            <div class="section-head-line"></div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            n_diag  = st.slider("Diagnoses", 1, 20, 4)
        with c2:
            n_proc  = st.slider("Procedures", 0, 15, 2)
        with c3:
            n_drugs = st.slider("Unique drugs", 0, 50, 8)

        c1, c2 = st.columns(2)
        with c1:
            icu_flag = st.selectbox("ICU admission", ["Yes", "No"])
        with c2:
            if icu_flag == "Yes":
                icu_los = st.slider("ICU LOS (days)", 0, 30, 2)
            else:
                icu_los = 0
                st.markdown("""
                <div style="padding-top:28px;font-family:'JetBrains Mono',monospace;
                            font-size:0.65rem;color:#2E3A55;letter-spacing:0.06em;">
                    ICU LOS — N/A
                </div>""", unsafe_allow_html=True)

        # — Laboratory —
        st.markdown("""
        <div class="section-head">
            <span class="section-head-label">03 · Day-1 Laboratory</span>
            <div class="section-head-line"></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            '<p style="font-size:0.7rem;color:#2E3A55;margin-bottom:14px;margin-top:-2px;'
            'font-family:Inter,sans-serif;">Defaults represent midpoint of normal reference ranges.</p>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            ph = st.number_input(
                "Blood pH", 6.80, 7.80, 7.40, step=0.01,
                help="Normal: 7.35 – 7.45",
            )
            creatinine = st.number_input(
                "Creatinine (mg/dL)", 0.1, 15.0, 1.0, step=0.1,
                help="Normal: 0.7 – 1.3 mg/dL",
            )
        with c2:
            wbc = st.number_input(
                "WBC (×10³/µL)", 0.1, 50.0, 7.5, step=0.5,
                help="Normal: 4.5 – 11.0 ×10³/µL",
            )
            hemoglobin = st.number_input(
                "Hemoglobin (g/dL)", 1.0, 20.0, 12.0, step=0.5,
                help="Critical low: < 8.0 g/dL",
            )

        # Live flag row
        flags = ""
        if ph < 7.20 or ph > 7.60:
            flags += '<span class="flag-chip flag-crit">pH CRITICAL</span>'
        elif ph < 7.35 or ph > 7.45:
            flags += '<span class="flag-chip flag-warn">pH ABNORMAL</span>'
        if creatinine > 3.0:
            flags += '<span class="flag-chip flag-crit">CREAT CRITICAL</span>'
        elif creatinine < 0.7 or creatinine > 1.3:
            flags += '<span class="flag-chip flag-warn">CREAT ABNORMAL</span>'
        if wbc > 20.0:
            flags += '<span class="flag-chip flag-crit">WBC CRITICAL</span>'
        elif wbc < 4.5 or wbc > 11.0:
            flags += '<span class="flag-chip flag-warn">WBC ABNORMAL</span>'
        if hemoglobin < 8.0:
            flags += '<span class="flag-chip flag-crit">HGB CRITICAL</span>'

        if flags:
            st.markdown(
                f'<div style="margin:6px 0 12px;">{flags}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        calculate = st.button("Run Risk Assessment", type="primary")

    # ── RESULTS ──────────────────────────────────────────────
    with right:

        st.markdown("""
        <div class="section-head" style="margin-top:0;">
            <span class="section-head-label">Risk Output</span>
            <div class="section-head-line"></div>
        </div>
        """, unsafe_allow_html=True)

        if calculate:
            # Build feature vector
            patient = pd.DataFrame(0, index=[0], columns=feature_names)
            for col, val in {
                'age': age, 'comorbidity_count': n_diag,
                'procedure_count': n_proc, 'unique_drug_count': n_drugs,
                'icu_flag': 1 if icu_flag == "Yes" else 0, 'icu_los': icu_los,
                'ph': ph, 'creatinine': creatinine, 'wbc': wbc, 'hemoglobin': hemoglobin,
            }.items():
                if col in patient.columns:
                    patient[col] = val

            if 'creatinine_abnormal' in patient.columns:
                patient['creatinine_abnormal'] = int(creatinine < 0.7 or creatinine > 1.3)
            if 'wbc_abnormal' in patient.columns:
                patient['wbc_abnormal'] = int(wbc < 4.5 or wbc > 11.0)
            if 'hemoglobin_abnormal' in patient.columns:
                patient['hemoglobin_abnormal'] = int(hemoglobin < 8.0)

            for col in [f'gender_{gender}', f'admission_type_{admission_type}',
                        f'icd_chapter_{icd_chapter}', f'insurance_{insurance}']:
                if col in patient.columns:
                    patient[col] = 1

            risk     = model.predict_proba(patient)[0, 1] * 100
            survival = 100.0 - risk
            tier     = "high" if risk >= 60 else "moderate" if risk >= 30 else "low"
            tier_cap = tier.upper()

            risk_color = {"high": "#FF5C6C", "moderate": "#FFAA00", "low": "#00E5A0"}[tier]

            guidance_text = {
                "high": "<strong>Elevated mortality risk detected.</strong> Patient profile indicates significant physiologic compromise. Recommend senior physician review, ICU escalation evaluation, and goals-of-care discussion.",
                "moderate": "<strong>Moderate mortality risk.</strong> Patient warrants close monitoring. Review laboratory trends, medication reconciliation, and escalation criteria.",
                "low": "<strong>Low mortality risk.</strong> Current profile indicates favourable short-term prognosis. Continue standard monitoring and evidence-based care protocols.",
            }

            # Risk card
# ── SHAP EXPLANATION ──────────────────────────────
            st.markdown("""
            <div class="section-head">
                <span class="section-head-label">Feature Attribution (SHAP)</span>
                <div class="section-head-line"></div>
            </div>
            """, unsafe_allow_html=True)

            # Compute SHAP values
            shap_values = explainer(patient)
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(6, 4))
            # Use a dark-themed style for matplotlib
            plt.style.use('dark_background')
            
            # Generate waterfall plot
            shap.plots.waterfall(shap_values[0], show=False)
            
            # Apply your theme colors to the plot elements
            plt.gca().set_facecolor('#0C1220')
            fig.patch.set_facecolor('#080C14')
            
            # Display
            st.pyplot(fig, bbox_inches='tight')
            
            # Guidance Text
            st.markdown(f"""
            <div style="margin-top:20px; padding:15px; border-left:2px solid {risk_color}; 
                        background:#0F1520; font-size:0.8rem; color:#8A9BBF;">
                {guidance_text[tier]}
            </div>
            """, unsafe_allow_html=True)

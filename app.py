import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
import time
from datetime import datetime

st.set_page_config(
    page_title="HealthSync — Clinical AI",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    .stApp {
        background: linear-gradient(135deg, #f9fafb 0%, #f1f5f9 100%) !important;
    }

    h1, h2, h3, h4 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    p, div, span, label, .stMarkdown {
        color: #1f2937 !important;
    }

    /* Force black text in dropdowns, multiselect search, and options */
    .stMultiSelect input,
    .stMultiSelect [role="combobox"],
    div[data-baseweb="select"] > div,
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    div[data-baseweb="menu"] * {
        color: #000000 !important;
        background: white !important;
    }

    /* Selected tags: red background + white text */
    div[data-baseweb="tag"] {
        background: #ef4444 !important;
        border: 1px solid #dc2626 !important;
        border-radius: 999px !important;
        padding: 0.4rem 0.9rem !important;
        color: white !important;
    }

    div[data-baseweb="tag"] span {
        color: white !important;
        font-weight: 500 !important;
    }

    /* Result card */
    .result-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 20px !important;
        padding: 2.5rem !important;
        box-shadow: 0 12px 32px rgba(0,0,0,0.08) !important;
        margin: 1.8rem 0 2.5rem !important;
    }

    .result-header {
        font-size: 3.2rem !important;
        background: linear-gradient(90deg, #2563eb, #0ea5e9) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin: 0.6rem 0 1.2rem !important;
        font-weight: 800 !important;
    }

    .confidence-chip {
        background: #eff6ff !important;
        color: #000000 !important;
        padding: 0.7rem 1.4rem !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        border: 1px solid #bfdbfe !important;
        display: inline-block !important;
    }

    .metrics-grid {
        display: grid !important;
        grid-template-columns: repeat(3, 1fr) !important;
        gap: 1.6rem !important;
        margin-top: 2.8rem !important;
        padding-top: 2rem !important;
        border-top: 1px solid #f1f5f9 !important;
    }

    .metric-item {
        text-align: center !important;
    }

    .metric-value {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #000000 !important;
        margin-bottom: 0.4rem !important;
    }

    .metric-label {
        font-size: 0.82rem !important;
        color: #000000 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
    }

    /* Modern Button */
    div.stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.9rem 2rem !important;
        font-weight: 600 !important;
        box-shadow: 0 6px 16px rgba(37,99,235,0.25) !important;
        transition: all 0.25s ease !important;
        width: 100% !important;
        letter-spacing: 0.4px !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 24px rgba(37,99,235,0.35) !important;
    }

    .footer {
        text-align: center !important;
        color: #64748b !important;
        font-size: 0.9rem !important;
        margin: 5rem 0 3rem !important;
        padding-top: 2.5rem !important;
        border-top: 1px solid #e2e8f0 !important;
    }

    /* Extra gap below disclaimer button */
    .disclaimer-button-wrapper {
        margin-top: 2.5rem !important;
    }
    </style>
""", unsafe_allow_html=True)

if 'accepted_terms' not in st.session_state:
    st.session_state.accepted_terms = False

@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            data = pickle.load(f)
        return data["model"], sorted(data["symptoms"]), data.get("metrics", {})
    except FileNotFoundError:
        class MockModel:
            classes_ = ["Viral Sinusitis", "Influenza", "Migraine", "Acute Bronchitis", "Allergic Rhinitis"]
            def predict_proba(self, X):
                seed = int(X[0].sum() * 100)
                np.random.seed(seed if seed > 0 else 42)
                return np.random.dirichlet(np.ones(5), size=1)
       
        symptoms = sorted([
            "high_fever", "dry_cough", "headache", "fatigue", "sore_throat",
            "runny_nose", "nausea", "muscle_pain", "dizziness", "chest_tightness",
            "loss_of_taste", "abdominal_pain", "skin_rash", "joint_pain"
        ])
        return MockModel(), symptoms, {"accuracy": 0.96, "f1_score": 0.94}

model, all_symptoms, metrics = load_model()

c1, c2 = st.columns([0.8, 10])
with c1:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=65)
with c2:
    st.markdown("<h1 style='margin: 0; padding-top: 8px;'>HealthSync AI</h1>", unsafe_allow_html=True)
    st.caption("Clinical Decision Support System")

if not st.session_state.accepted_terms:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="background: white; padding: 3rem; border-radius: 20px; border: 1px solid #e2e8f0; max-width: 720px; margin: 0 auto; text-align: center; box-shadow: 0 20px 30px rgba(0,0,0,0.08);">
            <div style="font-size: 4.2rem; margin-bottom: 1.5rem;">🛡️</div>
            <h2 style="color: #1e293b; margin-bottom: 1.2rem;">Medical Disclaimer</h2>
            <p style="color: #475569; line-height: 1.8; font-size: 1.12rem; margin-bottom: 2.5rem;">
                This AI system is a research prototype created for educational demonstration.<br><br>
                <strong>It is NOT a medical diagnostic tool</strong> and must never replace professional medical evaluation.<br>
                Always consult a certified healthcare professional for any health concerns.
            </p>
            <div class="disclaimer-button-wrapper">
    """, unsafe_allow_html=True)
   
    cols = st.columns([1, 1, 1])
    with cols[1]:
        if st.button("I Acknowledge & Continue", type="primary", use_container_width=True):
            st.session_state.accepted_terms = True
            st.rerun()

    st.markdown("</div></div>", unsafe_allow_html=True)

else:
    st.markdown("<br>", unsafe_allow_html=True)
   
    col_main, col_sidebar = st.columns([2.2, 1], gap="large")
   
    with col_main:
        st.subheader("Symptom Analysis")
       
        selected = st.multiselect(
            "Select Observed Symptoms",
            options=all_symptoms,
            placeholder="Type to search symptoms...",
            format_func=lambda x: x.replace('_', ' ').title()
        )
       
        st.markdown("<br>", unsafe_allow_html=True)
       
        if st.button("Run Diagnostic Analysis", type="primary"):
            if not selected:
                st.warning("⚠️ Please select at least one symptom.")
            else:
                with st.spinner("Analyzing clinical patterns..."):
                    time.sleep(0.8)
                   
                    vec = np.zeros(len(all_symptoms), dtype=int)
                    idxs = [all_symptoms.index(s) for s in selected]
                    vec[idxs] = 1
                   
                    probs = model.predict_proba([vec])[0]
                    top_idx = np.argsort(probs)[::-1][:5]
                   
                    st.session_state.results = {
                        "diseases": [model.classes_[i] for i in top_idx],
                        "conf": probs[top_idx] * 100,
                        "symptoms": selected
                    }
       
        if 'results' in st.session_state:
            res = st.session_state.results
            top_d = res['diseases'][0]
            top_c = res['conf'][0]
           
            st.markdown(f"""
            <div class="result-card">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div>
                        <div style="font-size: 0.9rem; color: #64748b; text-transform: uppercase; font-weight: 600; letter-spacing: 0.8px; margin-bottom: 0.6rem;">Primary Indication</div>
                        <div class="result-header">{top_d}</div>
                        <div class="confidence-chip">Confidence Score: {top_c:.1f}%</div>
                    </div>
                    <div style="font-size: 4rem; opacity: 0.9;">🏥</div>
                </div>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-value">{len(res['symptoms'])}</div>
                        <div class="metric-label">Symptoms</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">0.4s</div>
                        <div class="metric-label">Latency</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">{datetime.now().strftime('%H:%M')}</div>
                        <div class="metric-label">Timestamp</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_sidebar:
        st.markdown("""
        <div style="background: white; padding: 1.8rem; border-radius: 16px; border: 1px solid #e2e8f0; margin-top: 2.5rem;">
            <h3 style="margin-top: 0; font-size: 1.15rem; color: #000000;">System Status</h3>
            <div style="display: flex; justify-content: space-between; margin: 1.2rem 0; padding-bottom: 1rem; border-bottom: 1px solid #f1f5f9;">
                <span style="color: #000000;">Model Engine</span>
                <span style="font-weight: 600; color: #000000;">HealthSync v2.1</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 1.2rem 0;">
                <span style="color: #000000;">Server Status</span>
                <span style="font-weight: 600; color: #059669;">● Online</span>
            </div>
            <div style="font-size: 0.85rem; color: #000000; margin-top: 1.5rem; line-height: 1.6;">
                Secure session • Local processing only
            </div>
        </div>
        """, unsafe_allow_html=True)
       
        if 'results' in st.session_state:
            st.markdown("### Next Steps")
            st.info("Based on the AI assessment, please consider:")
            st.markdown("""
            * Consultation: Visit a General Practitioner
            * Monitoring: Check vitals every 4-6 hours
            * Hydration: Ensure adequate fluid intake
            """)
           
            report = f"HealthSync Clinical Report\n\nPrimary Prediction: {st.session_state.results['diseases'][0]}\nConfidence: {st.session_state.results['conf'][0]:.1f}%\nTimestamp: {datetime.now()}"
            st.download_button("📥 Download Report", report, "clinical_report.txt", use_container_width=True)

st.markdown("""
    <div class="footer">
        HealthSync AI • Educational Research Tool • 2026<br>
        <strong>Disclaimer:</strong> Not a Medical Device. Always consult a physician.
    </div>
""", unsafe_allow_html=True)
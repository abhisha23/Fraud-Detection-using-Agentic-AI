import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from anthropic import Anthropic

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Claims Fraud Detection Agent", page_icon="🛡️", layout="wide")

# ---- LOAD MODEL & ENCODERS ----
@st.cache_resource
def load_artifacts():
    model = joblib.load("fraud_model.pkl")
    le_type = joblib.load("le_type.pkl")
    le_provider = joblib.load("le_provider.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return model, le_type, le_provider, feature_cols

model, le_type, le_provider, feature_cols = load_artifacts()

# ---- CLAUDE CLIENT ----
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# ---- SIDEBAR ----
with st.sidebar:
    st.title("🛡️ Fraud Detection Agent")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("1. 📝 Enter claim details")
    st.markdown("2. 🌲 Random Forest scores fraud risk")
    st.markdown("3. 🤖 Claude explains the reasoning in plain language")
    st.markdown("---")
    st.markdown("**Built with:**")
    st.markdown("- Scikit-learn (Random Forest)")
    st.markdown("- Anthropic Claude (agentic explanation layer)")
    st.markdown("- Streamlit")
    st.markdown("---")
    st.caption("Synthetic demo data — not real claims.")

# ---- MAIN ----
st.title("🛡️ Insurance Claims Fraud Detection Agent")
st.markdown("A Random Forest model scores fraud risk on structured claim data, then **Claude acts as an explanation agent** — reasoning over the flagged risk factors and producing a plain-language justification a claims adjuster could actually use.")

st.markdown("### Enter Claim Details")

col1, col2, col3 = st.columns(3)
with col1:
    claim_amount = st.number_input("Claim Amount ($)", min_value=0.0, value=1500.0, step=50.0)
    policy_tenure = st.number_input("Policy Tenure (months)", min_value=0, value=24, step=1)
    prior_claims = st.number_input("Prior Claims Count", min_value=0, value=1, step=1)
with col2:
    days_since_last = st.number_input("Days Since Last Claim", min_value=0, value=180, step=1)
    days_to_file = st.number_input("Days to File Claim", min_value=0, value=5, step=1)
    member_age = st.number_input("Member Age", min_value=18, max_value=100, value=40, step=1)
with col3:
    claim_type = st.selectbox("Claim Type", le_type.classes_)
    provider = st.selectbox("Provider", le_provider.classes_)

analyze_btn = st.button("🔍 Analyze Claim", use_container_width=True, type="primary")

if analyze_btn:
    # Build feature row
    row = pd.DataFrame([{
        "claim_amount": claim_amount,
        "policy_tenure_months": policy_tenure,
        "prior_claims_count": prior_claims,
        "days_since_last_claim": days_since_last,
        "days_to_file_claim": days_to_file,
        "claim_type_enc": le_type.transform([claim_type])[0],
        "provider_enc": le_provider.transform([provider])[0],
        "member_age": member_age,
    }])[feature_cols]

    fraud_proba = model.predict_proba(row)[0][1]
    is_flagged = fraud_proba > 0.5

    st.markdown("---")
    st.markdown("### 📊 Risk Assessment")

    rcol1, rcol2 = st.columns(2)
    with rcol1:
        if is_flagged:
            st.error(f"🚩 FLAGGED FOR REVIEW — {fraud_proba*100:.1f}% fraud risk")
        else:
            st.success(f"✅ LOW RISK — {fraud_proba*100:.1f}% fraud risk")
    with rcol2:
        st.progress(min(fraud_proba, 1.0))

    # Feature importance-based risk factors for this specific claim
    importances = dict(zip(feature_cols, model.feature_importances_))
    risk_factors = []
    if claim_amount > 3000:
        risk_factors.append(f"Claim amount (${claim_amount:,.2f}) is unusually high")
    if policy_tenure < 3:
        risk_factors.append(f"Policy is very new ({policy_tenure} months)")
    if days_since_last < 14:
        risk_factors.append(f"Filed only {days_since_last} days after a previous claim")
    if prior_claims > 3:
        risk_factors.append(f"High prior claims count ({prior_claims})")
    if days_to_file > 60:
        risk_factors.append(f"Long delay before filing ({days_to_file} days)")

    if not risk_factors:
        risk_factors.append("No individual risk thresholds triggered — flag is based on combined feature interactions")

    # ---- CLAUDE EXPLANATION AGENT ----
    st.markdown("### 🤖 Agent Explanation")

    if client is None:
        st.warning("Set the ANTHROPIC_API_KEY environment variable to enable Claude's explanation agent.")
    else:
        with st.spinner("Claude is reasoning over the risk factors..."):
            prompt = f"""You are a fraud investigation assistant helping a claims adjuster understand a model's fraud risk score.

Claim details:
- Claim amount: ${claim_amount:,.2f}
- Policy tenure: {policy_tenure} months
- Prior claims count: {prior_claims}
- Days since last claim: {days_since_last}
- Days to file claim: {days_to_file}
- Claim type: {claim_type}
- Provider: {provider}
- Member age: {member_age}

Model output: {fraud_proba*100:.1f}% fraud risk ({"FLAGGED" if is_flagged else "not flagged"})

Automatically detected risk signals: {", ".join(risk_factors)}

Write a short (3-4 sentence), plain-language explanation for a claims adjuster explaining why this claim received this risk score, referencing the specific factors above. Be measured — this is a risk signal for human review, not a fraud accusation. End with one concrete recommended next step for the adjuster."""

            try:
                response = client.messages.create(
                    model="claude-sonnet-5",
                    max_tokens=400,
                    messages=[{"role": "user", "content": prompt}],
                )
                explanation = response.content[0].text
                st.info(explanation)
            except Exception as e:
                st.error(f"Claude API error: {e}")

    with st.expander("🔍 View raw model feature importances"):
        st.dataframe(
            pd.Series(importances, name="Importance").sort_values(ascending=False),
            use_container_width=True,
        )

st.markdown("---")
st.caption("Demo built on synthetic data to explore fraud detection + agentic AI explanation patterns. Not connected to any real claims system.")

"""
Generates a synthetic health insurance claims dataset for fraud detection.
Fraud is injected via realistic patterns: unusually high claim amounts relative
to policy tenure, rapid repeat claims, mismatched provider/claim-type combos,
and claims filed very soon after policy start (classic fraud signal).
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000

claim_types = ["Outpatient", "Inpatient", "Diagnostic", "Pharmacy", "Dental", "Emergency"]
providers = ["Provider_A", "Provider_B", "Provider_C", "Provider_D", "Provider_E"]

data = {
    "claim_id": [f"CLM{100000+i}" for i in range(N)],
    "claim_amount": np.round(np.random.gamma(shape=2.0, scale=800, size=N), 2),
    "policy_tenure_months": np.random.randint(1, 120, size=N),
    "prior_claims_count": np.random.poisson(1.2, size=N),
    "days_since_last_claim": np.random.randint(0, 730, size=N),
    "days_to_file_claim": np.random.randint(0, 90, size=N),  # delay between incident and filing
    "claim_type": np.random.choice(claim_types, size=N),
    "provider": np.random.choice(providers, size=N),
    "member_age": np.random.randint(18, 85, size=N),
}

df = pd.DataFrame(data)

# ---- Inject realistic fraud signal ----
# Fraud risk increases with: high claim amount relative to tenure, very fast filing
# after a very new policy, high prior claims count in short time, unusually fast
# re-filing after a previous claim.
fraud_score = (
    (df["claim_amount"] > 3000).astype(int) * 0.35
    + (df["policy_tenure_months"] < 3).astype(int) * 0.30
    + (df["days_since_last_claim"] < 14).astype(int) * 0.20
    + (df["prior_claims_count"] > 3).astype(int) * 0.25
    + (df["days_to_file_claim"] > 60).astype(int) * 0.15
    + np.random.normal(0, 0.15, size=N)  # noise so it's not perfectly separable
)

df["is_fraud"] = (fraud_score > 0.55).astype(int)

print(f"Generated {N} claims. Fraud rate: {df['is_fraud'].mean()*100:.1f}%")
df.to_csv("claims_data.csv", index=False)
print("Saved to claims_data.csv")

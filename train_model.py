import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import joblib

df = pd.read_csv("claims_data.csv")

# Encode categoricals
le_type = LabelEncoder()
le_provider = LabelEncoder()
df["claim_type_enc"] = le_type.fit_transform(df["claim_type"])
df["provider_enc"] = le_provider.fit_transform(df["provider"])

feature_cols = [
    "claim_amount", "policy_tenure_months", "prior_claims_count",
    "days_since_last_claim", "days_to_file_claim", "claim_type_enc",
    "provider_enc", "member_age",
]

X = df[feature_cols]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

print("\nFeature Importances:")
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importances)

# Save model + encoders
joblib.dump(model, "fraud_model.pkl")
joblib.dump(le_type, "le_type.pkl")
joblib.dump(le_provider, "le_provider.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")
print("\nModel and encoders saved.")

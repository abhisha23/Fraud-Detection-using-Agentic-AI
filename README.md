# 🛡️ Insurance Claims Fraud Detection Agent

An AI-powered fraud detection system that combines a traditional machine learning classifier with an LLM-based agentic reasoning layer — built to explore how structured ML predictions and generative AI can work together in a real-world insurance workflow.


## What it does

1. A claims adjuster (or automated intake system) enters structured claim details — amount, policy tenure, prior claims history, filing delay, claim type, provider, and member age.
2. A **Random Forest classifier** scores the claim's fraud risk based on patterns learned from historical claims data.
3.Claude (Anthropic)acts as an agentic explanation layer — it takes the model's risk score and the specific triggered risk factors, reasons over them, and produces a clear, plain-language justification a human adjuster can actually act on, along with a recommended next step.

The goal is a system where the ML model handles pattern detection at scale, and the LLM agent handles the human-facing reasoning and communication — each doing what it's best at.

---

## Architecture

```
Claim Input (Streamlit form)
        │
        ▼
Random Forest Classifier ──► Fraud Risk Score (%)
        │
        ▼
Risk Factor Detection (rule-based thresholds)
        │
        ▼
Claude API (agentic reasoning) ──► Plain-language explanation + recommended action
        │
        ▼
Displayed to the adjuster in the UI
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| ML Model | Scikit-learn (Random Forest, class-balanced) |
| Agentic Reasoning | Anthropic Claude API |
| Frontend | Streamlit |
| Data | Synthetic claims dataset (5,000 records, ~3% fraud rate) |

---

## Why these choices?

- **Random Forest** — robust to the class imbalance inherent in real fraud detection (fraud is rare), and its feature importances give interpretable risk signals to feed into the explanation layer.
- **Claude as the reasoning layer** — rather than hard-coding explanation templates, an LLM can synthesize the specific combination of risk factors for *this* claim into a coherent, adjuster-ready narrative, and adapt its tone/recommendation to the specific case.
- **Synthetic data** — built with realistic, documented fraud signals (high claim amount relative to tenure, rapid repeat filing, high prior claims count, filing delays) so the model's behavior is explainable and the dataset carries no real personal information.

---

## Model Performance

- **ROC-AUC:** 0.94
- Fraud prevalence in training data: ~3.1% (intentionally imbalanced to mirror real-world fraud rates)
- Class-balanced training to avoid the model simply predicting "not fraud" for everything

---

## Running Locally

```bash
# Clone the repo
git clone https://github.com/abhisha23/Fraud-Detection-using-Agentic-AI.git
cd Fraud-Detection-using-Agentic-AI

# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-key-here"

# (Optional) Regenerate the dataset and retrain the model
python generate_data.py
python train_model.py

# Run the app
streamlit run app.py
```

---

## Project Files

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application |
| `generate_data.py` | Generates the synthetic claims dataset with injected fraud patterns |
| `train_model.py` | Trains and evaluates the Random Forest classifier |
| `fraud_model.pkl` | Trained model artifact |
| `le_type.pkl`, `le_provider.pkl` | Label encoders for categorical features |
| `feature_cols.pkl` | Feature column order used by the model |
| `claims_data.csv` | Synthetic training dataset |
| `requirements.txt` | Python dependencies |

---

## Disclaimer

This project uses entirely **synthetic data** for demonstration purposes. It is not connected to any real claims system, insurer, or individual's data, and is not intended for production fraud detection use without significant additional validation, bias testing, and regulatory review.

---

## Author

Abirami Shanmugam — AI/ML Engineer
[GitHub](https://github.com/abhisha23) 

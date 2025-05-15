import os
from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# ─── Configuration ───
MODEL_REPO = "Promitsaha1/best_model_LLM_annotation"
# Force the slow tokenizer to avoid tiktoken issues
tokenizer  = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=False)
model      = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)

# Temperature scaling factor (1.0 = no scaling)
TEMPERATURE = float(os.environ.get("TEMPERATURE", 1.0))

# Your four biases and their per-label thresholds
LABEL_COLS = [
    "Anchoring",
    "Illusory Truth Effect",
    "Information Overload",
    "Mere-Exposure Effect"
]
THRS = {
    "Anchoring":                 0.65,
    "Illusory Truth Effect":     0.65,
    "Information Overload":      0.65,
    "Mere-Exposure Effect":      0.65
}

def compute_phishing_risk(body: str):
    # 1) Tokenize + model inference
    inputs = tokenizer(
        body,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits  # shape (1,4)

    # 2) Apply temperature scaling
    scaled_logits = logits / TEMPERATURE

    # 3) Sigmoid → probabilities
    probs = torch.sigmoid(scaled_logits).squeeze().tolist()  # [p0,p1,p2,p3]

    # 4) Risk score = max probability × 100
    risk_pct = max(probs) * 100

    # 5) Triggered labels using per-label thresholds
    triggered = [
        LABEL_COLS[i]
        for i, p in enumerate(probs)
        if p >= THRS[LABEL_COLS[i]]
    ]

    return risk_pct, dict(zip(LABEL_COLS, probs)), triggered

@app.route("/", methods=["GET", "POST"])
def index():
    # Default context
    context = {
        "body":      "",
        "risk":      None,
        "probs":     None,
        "triggered": None
    }

    if request.method == "POST":
        body = request.form["body"]
        risk, probs, triggered = compute_phishing_risk(body)
        context.update({
            "body":      body,
            "risk":      f"{risk:.1f}",
            "probs":     {k: f"{v:.3f}" for k, v in probs.items()},
            "triggered": triggered
        })

    return render_template("index.html", **context)

if __name__ == "__main__":
    # Use the PORT env var for Railway compatibility
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

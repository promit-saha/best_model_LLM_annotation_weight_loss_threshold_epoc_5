import os
from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# ─── Load model + tokenizer ─────────────────────────────────────────────────
MODEL_REPO = "Promitsaha1/best_model_LLM_annotation_weight_loss_threshold_epoc_5"
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
model.eval()

# ─── Labels and their trigger thresholds ────────────────────────────────────
LABEL_COLS = [
    "Anchoring",
    "Illusory Truth Effect",
    "Information Overload",
    "Mere-Exposure Effect"
]
THRESHOLDS = {
    "Anchoring":               0.60,
    "Illusory Truth Effect":   0.65,
    "Information Overload":    0.60,
    "Mere-Exposure Effect":    0.60
}

def compute_phishing_risk(body: str):
    # 1) Tokenize + infer
    inputs = tokenizer(
        body,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits).squeeze().tolist()

    # 2) Map labels ↔ probabilities
    probs_dict = dict(zip(LABEL_COLS, probs))

    # 3) Find which cues exceed their threshold
    triggered = [
        lbl for lbl, p in probs_dict.items()
        if p >= THRESHOLDS[lbl]
    ]

    # 4) Risk = average probability of only the triggered cues
    if triggered:
        risk_pct = sum(probs_dict[l] for l in triggered) / len(triggered) * 100
    else:
        risk_pct = 0.0

    return risk_pct, probs_dict, triggered

@app.route("/", methods=["GET", "POST"])
def index():
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
            "risk":      f"{risk:.1f}%",                # e.g. "73.6%"
            "probs":     {k: f"{v:.3f}" for k, v in probs.items()},
            "triggered": triggered
        })

    return render_template("index.html", **context)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

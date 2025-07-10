from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    DistilBertTokenizerFast, DistilBertForSequenceClassification
)

# Setup
app = Flask(__name__)
CORS(app)  # Enable cross-origin for frontend access

# Load GPT-2
print("🔄 Loading GPT-2...")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.eval()

# Load DistilBERT
print("🔄 Loading DistilBERT...")
bert_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
bert_model.eval()

# Generate Fake News
def generate_fake_news(prompt):
    input_ids = gpt2_tokenizer.encode(prompt, return_tensors="pt")
    output = gpt2_model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=1,
        temperature=0.9,
        top_p=0.95,
        do_sample=True,
        no_repeat_ngram_size=2
    )
    return gpt2_tokenizer.decode(output[0], skip_special_tokens=True)

# Detect Real/Fake
def detect_news(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item()
    label = "REAL" if pred == 1 else "FAKE"
    return label, round(confidence * 100, 2)

# ========== API ROUTES ==========

@app.route("/generate", methods=["POST"])
def api_generate():
    prompt = request.json.get("prompt", "")
    generated = generate_fake_news(prompt)
    return jsonify({"generated": generated})

@app.route("/detect", methods=["POST"])
def api_detect():
    text = request.json.get("text", "")
    label, confidence = detect_news(text)
    return jsonify({"label": label, "confidence": confidence})

@app.route("/generate_and_detect", methods=["POST"])
def api_generate_and_detect():
    prompt = request.json.get("prompt", "")
    generated = generate_fake_news(prompt)
    label, confidence = detect_news(generated)
    return jsonify({"generated": generated, "label": label, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)

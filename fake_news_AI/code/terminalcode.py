import torch
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    DistilBertTokenizerFast, DistilBertForSequenceClassification
)
import warnings
warnings.filterwarnings("ignore")

# ============ Load Models ============
print("🔄 Loading models (GPT-2 and DistilBERT)...")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.eval()

# Pretrained sentiment model used as placeholder for fake/real detection
bert_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
bert_model.eval()

# ============ GPT-2: Fake News Generator ============
def generate_fake_news(prompt="Breaking News:"):
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

# ============ DistilBERT: News Detector ============
def detect_news(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item()

    # Use sentiment polarity as fake/real (only for demo)
    # class 0: NEGATIVE → FAKE, class 1: POSITIVE → REAL
    label = "REAL" if pred == 1 else "FAKE"
    return label, confidence

# ============ Console Menu ============
def menu():
    print("\n Welcome to Fake News Generator & Detector")
    print("1. Generate Fake News using GPT-2")
    print("2. Detect Fake/Real using BERT")
    print("3. Generate & Detect")
    print("4. Exit")

# ============ Main Loop ============
while True:
    menu()
    choice = input("Choose an option (1/2/3/4): ").strip()

    if choice == "1":
        prompt = input("\n Enter a topic or headline start: ")
        generated = generate_fake_news(prompt)
        print("\n Generated Fake News Headline:\n", generated)

    elif choice == "2":
        text = input("\n Enter a news headline to detect: ")
        label, conf = detect_news(text)
        print(f"\n Detection Result: {label} (Confidence: {conf*100:.2f}%)")

    elif choice == "3":
        prompt = input("\n Enter a topic or headline start: ")
        generated = generate_fake_news(prompt)
        print("\n Generated Headline:\n", generated)
        label, conf = detect_news(generated)
        print(f"\n Detection Result: {label} (Confidence: {conf*100:.2f}%)")

    elif choice == "4":
        print("\n Exiting. Thank you for using the tool!")
        break
    else:
        print("Invalid choice. Please enter 1, 2, 3 or 4.")
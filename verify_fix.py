import joblib
import os
import re

# Load Models
BASE_DIR = os.getcwd()
sentiment_model = joblib.load(os.path.join(BASE_DIR, 'sentiment_model.pkl'))
issue_model = joblib.load(os.path.join(BASE_DIR, 'issue_model.pkl'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

test_cases = [
    ("I hate this product, it is terrible.", "negative"),
    ("We are so happy you liked it!", "positive"),
    ("It is okay, nothing special.", "neutral"),
]

print("--- Verification Results ---")
all_passed = True
for text, expected in test_cases:
    cleaned = clean_text(text)
    prediction = sentiment_model.predict([cleaned])[0]
    print(f"Input: '{text}' | Prediction: {prediction} | Expected: {expected}")
    if prediction != expected:
        all_passed = False

if all_passed:
    print("\nSUCCESS: All test cases passed.")
else:
    print("\nFAILURE: Some test cases failed.")

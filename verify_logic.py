import joblib
import re
import random
import os

# --- LOGIC TO TEST (COPIED FROM APP.PY) ---

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def classify_intent_hybrid(text, sentiment_model, issue_model):
    text_clean = clean_text(text)
    
    # Rule-based Overrides
    if any(w in text_clean for w in ['refund', 'money back', 'charge', 'cost', 'price']):
        return "negative", "Billing"
    if any(w in text_clean for w in ['broken', 'damaged', 'defective', 'stopped working', 'quality', 'bad', 'worst']):
        return "negative", "Quality"
    if any(w in text_clean for w in ['late', 'delivery', 'arrive', 'shipping', 'package']):
        return "negative", "Logistics"
    if any(w in text_clean for w in ['amazing', 'love', 'great', 'best', 'happy', 'good']):
        return "positive", "General"
    
    # Fallback to AI Model
    pred_sentiment = sentiment_model.predict([text_clean])[0]
    pred_issue = issue_model.predict([text_clean])[0]
    
    return pred_sentiment, pred_issue

def generate_response(sentiment, issue, text):
    text_clean = clean_text(text)
    
    responses = {
        "negative": {
            "Logistics": [
                "I see you're facing a delivery issue. I've prioritized this with our Logistics Team. Do you have your Order ID?",
                "I apologize for the delay. Shipping times have been inconsistent lately. Let me check the status for you immediately.",
                "That is frustrating. I can track that package for you right now. Please share your Order ID."
            ],
            "Quality": [
                "I am truly sorry to hear the product is defective. We have a 100% replacement policy. would you like a refund or a new unit?",
                "That's not the quality we strive for. I've flagged this for a Return Request. You'll receive a confirmation email shortly.",
                "Oh no, I apologize. Since the item is damaged, we can ship a replacement immediately. completely free of charge."
            ],
            "Billing": [
                "I understand your concern about the charges. I'm connecting you to our Finance specialist to sort this out.",
                " refunds are usually processed within 24 hours. Let me initiate that for you right now.",
                "I see the billing discrepancy. I'll open a ticket with Finance to get your money back."
            ],
            "General": [
                "I'm sorry you're having a bad experience. Can you tell me more about what went wrong?",
                "I apologize if we let you down. A senior support agent will be with you in moment.",
                "Thank you for bringing this to our attention. We take this feedback very seriously."
            ]
        },
        "positive": {
            "General": [
                "We are thrilled to hear that! 🎉 Here is a special 10% discount code for your next order: HAPPY10",
                "That's music to our ears! Thank you for being a loyal customer.",
                "Wow, thank you! We'll make sure to pass your kind words to the team."
            ]
        },
        "neutral": {
            "General": [
                "Thanks for the update. Is there anything else I can help you with today?",
                "Understood. Let us know if you have more questions.",
                "Okay, I've noted that down. Anything else?"
            ],
            "Logistics": [
                "Your order is on the way. You should receive a tracking link via email soon.",
                "Standard delivery usually takes 3-5 business days."
            ]
        }
    }
    
    # Context Awareness
    if re.search(r'\b\d{5,}\b', text):
            return "Thanks for providing the Order ID. I am pulling up your details now... ⏳"

    try:
        category_responses = responses.get(sentiment, {}).get(issue, responses["neutral"]["General"])
        return random.choice(category_responses)
    except:
        return "I'm here to help. Could you provide more details?"

# --- RUNNER ---

# Load Models (Must be in same directory)
try:
    print("Loading models...")
    sent_model = joblib.load('sentiment_model.pkl')
    iss_model = joblib.load('issue_model.pkl')
except:
    print("ERROR: Models not found in execution directory.")
    exit()

test_cases = [
    "product is damaged",
    "delivery is late",
    "amazing experience",
    "bad product",
    "i need a refund",
    "Order #12345 is missing",
    "It's okay, nothing special",
    "tracking not working",
    "I hate this service"
]

print("\n--- Running Verification Suite ---\n")

for text in test_cases:
    sent, issue = classify_intent_hybrid(text, sent_model, iss_model)
    resp = generate_response(sent, issue, text)
    
    print(f"User Input: '{text}'")
    print(f"   -> Detected: {sent.upper()} | {issue}")
    print(f"   -> Bot Says: \"{resp}\"")
    print("-" * 50)

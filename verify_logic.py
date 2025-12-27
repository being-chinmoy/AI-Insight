import joblib
import re
import random
import os

# --- LOGIC TO TEST (COPIED FROM APP.PY) ---

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    
    # Typo Correction (Basic manual list for common support terms)
    typos = {
        'replcaement': 'replacement',
        'replacment': 'replacement',
        'refunds': 'refund',
        'exchange': 'replacement',
        'mismtch': 'mismatch',
        'colr': 'color',
        'colour': 'color'
    }
    for typo, fix in typos.items():
        text = text.replace(typo, fix)
        
    return text

def classify_intent_hybrid(text, sentiment_model, issue_model):
    text_clean = clean_text(text)
    
    # Rule 0: Gibberish / Very Short Input Check
    short_sentiment_words = ['bad', 'sad', 'mad', 'wow', 'hate', 'good', 'love', 'best', 'ok', 'okay', 'poor', 'nice', 'help']
    if len(text_clean) < 6 and text_clean not in short_sentiment_words:
        if text_clean in ['hi', 'hello', 'hey', 'thanks', 'thank', 'thx', 'refund']: # Added refund to known safe words
            return "neutral", "General"
        return "neutral", "General"

    # Rule 1: Positive Keywords
    if 'fast' in text_clean and 'delivery' in text_clean:
            return "positive", "Logistics"
    if any(w in text_clean for w in ['amazing', 'love', 'great', 'best', 'happy', 'good', 'excellent']):
            return "positive", "General"

    # Rule 2: Negative Specific Logic
    if any(w in text_clean for w in ['email', 'contact', 'support', 'human', 'agent', 'help desk']):
        return "neutral", "Support"
    if any(w in text_clean for w in ['refund', 'money back', 'charge', 'cost', 'price', 'expensive', 'worth']):
        return "negative", "Billing"
    if any(w in text_clean for w in ['broken', 'damaged', 'defective', 'working', 'quality', 'bad', 'mismatch', 'color', 'wrong']):
        return "negative", "Quality"
    if any(w in text_clean for w in ['late', 'delivery', 'arrive', 'shipping', 'package', 'missing', 'lost', 'where']):
        return "negative", "Logistics"
    
    # Fallback to AI Model
    pred_sentiment = sentiment_model.predict([text_clean])[0]
    pred_issue = issue_model.predict([text_clean])[0]
    
    return pred_sentiment, pred_issue

def generate_response(sentiment, issue, text, rating):
    text_clean = clean_text(text)
    
    # Critical Priority: Gratitude check
    if text_clean.startswith("thank") or "thank you" in text_clean or "thanks" in text_clean:
         return "You are very welcome! Let me know if you need anything else."

    # Context Awareness
    if re.search(r'\b\d{5,}\b', text):
            return "Thanks for providing the Order ID. I am pulling up your details now... ⏳"

    # --- RATING BASED LOGIC FLOW ---
    
    # 1. Low Rating (0-1: Stars 1-2)
    if rating is not None and rating <= 1:
        # CHECK FOR USER CHOICE FIRST (Resolution)
        if "refund" in text_clean:
            return "✅ **Refund Initiated.** You will receive the amount in your original payment method within 3-5 business days."
        if "replacement" in text_clean:
            return "📦 **Replacement Order Created.** We will ship the new item immediately. You can keep the old one or discard it."

        if issue == "Quality":
            return "I apologize for the poor quality/mismatch. Since you rated us low, I can offer an immediate **Refund** or **Replacement**. Which do you prefer?"
        elif issue == "Billing":
            return "I see you feel the price is unfair. I can initiate a **Full Refund** right now."
        elif issue == "Logistics":
            return "I'm sorry for the shipping trouble. I can process a refund for the shipping charges immediately."
    
    # 2. Mid Rating (2: Star 3)
    if rating is not None and rating == 2:
        if "email" in text_clean or "help" in text_clean or "desk" in text_clean or "connect" in text_clean:
                return "I have registered your request. 📧 **Sending an escalation email to support@sportsphere.com...** [Sent Successfully]"

    # 3. High Rating (3-4: Stars 4-5)
    if rating is not None and rating >= 3:
             # Dynamic acknowledgment
             if "delivery" in text_clean:
                 return "Glad to hear the **Delivery** was up to the mark! 🚚💨 Thanks for the 5 stars!"
             if "quality" in text_clean or "product" in text_clean:
                 return "Awesome! We pride ourselves on **Component Quality**. Enjoy your gear! 🏆"
             if "price" in text_clean:
                 return "We try to offer the best value. Thanks for noticing! 💸"
             
             return "We are over the moon! 🌙 Thank you for your support. Keep rocking!"

    # Default Logic
    responses = {
        "negative": {
            "Logistics": ["Tracking shows a delay. I'm sorry."],
            "Quality": ["That sounds frustrating. We can swap that item for you."],
            "Billing": ["Let me check why you were charged that amount."],
            "Support": ["Here is our support email: support@sportsphere.com."],
            "General": ["I apologize for the bad experience."]
        },
        "neutral": {
            "Support": ["Our help desk is available 24/7 at support@sportsphere.com."],
            "General": ["How else can I help?"]
        },
        "positive": {
            "Logistics": ["Glad the delivery was fast!"],
            "General": ["Thanks!"]
        }
    }
    
    try:
        category_responses = responses.get(sentiment, {}).get(issue, responses["neutral"]["General"])
        return random.choice(category_responses)
    except:
        return "I'm listening. Tell me more."

# --- RUNNER ---
try:
    print("Loading models...")
    sent_model = joblib.load('sentiment_model.pkl')
    iss_model = joblib.load('issue_model.pkl')
except:
    print("ERROR: Models not found in execution directory.")
    exit()

print("\n--- Running Logic Verification ---\n")

# Format: (Input Text, User Rating 0-4 (None means no rating), Expected Description)
tests = [
    ("fast delivery", None, "Should be POSITIVE Logistics"),
    ("product is missing", None, "Should be NEGATIVE Logistics"),
    ("colour is mismatch", 0, "Should auto-offer REFUND/REPLACEMENT (Rating 1)"),
    ("price is high", 1, "Should auto-offer REFUND (Rating 2)"),
    ("connect to help desk", 2, "Should SEND EMAIL (Rating 3)"),
    ("amazing experience", 4, "Should be OVER THE MOON (Rating 5)"),
    ("dsdsv", None, "Should be Neutral (Gibberish)"),
    ("thank you", 0, "Should say WELCOME (Rating 1) - fix for apology bug"),
    ("replcaement", 0, "Should say REPLACEMENT ORDER CREATED (Typo fix)"),
    ("Refund", 0, "Should say REFUND INITIATED"),
    ("delivery", 4, "Should say Glad to hear Delivery was good (Rating 5)"),
    ("billing", 4, "Should say Thanks for support (Rating 5) - generic")
]

for text, rating, desc in tests:
    sent, issue = classify_intent_hybrid(text, sent_model, iss_model)
    resp = generate_response(sent, issue, text, rating)
    
    rating_display = f"{rating+1} Stars" if rating is not None else "No Rating"
    
    print(f"Test: '{text}' [{rating_display}]")
    print(f"   Context: {desc}")
    print(f"   -> Detected: {sent.upper()} | {issue}")
    print(f"   -> Bot Says: \"{resp}\"")
    print("-" * 50)

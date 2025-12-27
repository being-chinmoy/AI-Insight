import joblib
import re
import os

# 1. Load Both Models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    sentiment_model = joblib.load(os.path.join(BASE_DIR, 'sentiment_model.pkl'))
    issue_model = joblib.load(os.path.join(BASE_DIR, 'issue_model.pkl'))
    print("System: Models loaded successfully.")
except FileNotFoundError:
    print("Error: Models not found. Run 'train_model.py' first.")
    exit()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def get_chat_response(user_text):
    cleaned_blob = clean_text(user_text)
    # Model A Prediction
    sentiment = sentiment_model.predict([cleaned_blob])[0]
    
    # Model B Prediction
    issue_category = issue_model.predict([cleaned_blob])[0]
    
    # Logic Engine
    response = ""
    
    if sentiment == 'negative':
        if issue_category == 'Logistics':
            response = "I see you are facing a delivery issue. I have escalated this to our Logistics Team (Priority: High)."
        elif issue_category == 'Quality':
            response = "I apologize for the defective product. I have initiated a Return Request for you immediately."
        elif issue_category == 'Billing':
            response = "I understand your concern about the price/refund. Connecting you to the Finance Department."
        else:
            response = "I'm sorry you're having trouble. A support agent will contact you shortly."
            
    elif sentiment == 'positive':
        response = "We are so happy you liked it! Here is a 10% discount code: HAPPY10"
        
    else: # Neutral
        if issue_category == 'Logistics':
            response = "Your order is currently in transit and should arrive by tomorrow."
        else:
            response = "Thank you for your feedback. Let us know if we can help with anything else."
            
    return sentiment, issue_category, response

# Chat Loop
print("\n--- AI Customer Support (Type 'quit' to exit) ---")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == 'quit': break
    
    sent, issue, reply = get_chat_response(user_input)
    
    print(f"   [Analyzed: Sentiment={sent}, Issue={issue}]")
    print(f"Bot: {reply}")
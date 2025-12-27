import pandas as pd
import re
import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. Load Data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'Customer_Sentiment.csv')

def load_and_augment_data():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.DataFrame(columns=['review_text', 'sentiment'])
    
    # --- 1.1 Extensive Data Augmentation ---
    # We need to ensure the model understands specific short phrases and high-emotion words.
    print("   > Augmenting data with ADVANCED synthetic examples...")
    
    # Advanced Synthetic Data Generation
    # Combining intents with varied phrasings
    synthetic_data = []

    # Negative Categories
    logistics_issues = [
        "late delivery", "shipping is too slow", "never arrived", "tracking not working",
        "where is my package", "courier was rude", "delivery delayed", "package lost",
        "bad packaging", "box crushed", "damaged in transit", "took forever to arrive"
    ]
    quality_issues = [
        "product is damaged", "broken item", "stopped working", "defective unit",
        "poor quality", "worst product ever", "useless", "cheap material",
        "falling apart", "not as described", "different from picture", "malfunctioned"
    ]
    billing_issues = [
        "refund needed", "cancel my order", "too expensive", "not worth the money",
        "overcharged", "hidden fees", "billing error", "credit card charged twice",
        "want my money back", "price is high", "fraudulent charge", "invoice incorrect"
    ]
    support_issues = [
        "customer service is bad", "rude support", "no reply", "ignored my emails",
        "unhelpful agent", "terrible support", "waiting on hold forever", "bot is useless"
    ]

    # Positive & Neutral
    positives = [
        "amazing experience", "love it", "great product", "highly recommend",
        "super fast", "exceptional quality", "worth every penny", "best purchase",
        "very happy", "fantastic service", "five stars", "wonderful"
    ]
    neutrals = [
        "it's okay", "average", "nothing special", "delivered today",
        "works fine", "acceptable", "generic product", "as expected",
        "just arrived", "no complaints so far", "mediocre"
    ]

    # Helper to add variations
    def add_entries(texts, sentiment, multiplier=50):
        for text in texts:
            # Lowercase version
            synthetic_data.append({"review_text": text.lower(), "sentiment": sentiment})
            # Original version (for some casing variety if vectorizer cares, though we clean it later)
            synthetic_data.append({"review_text": text, "sentiment": sentiment})
            # Upper case for emphasis
            synthetic_data.append({"review_text": text.upper(), "sentiment": sentiment})

    # Add massively to ensure these patterns stick
    add_entries(logistics_issues + quality_issues + billing_issues + support_issues, "negative", multiplier=100)
    add_entries(positives, "positive", multiplier=100)
    add_entries(neutrals, "neutral", multiplier=100)

    # Convert to DataFrame
    aug_df = pd.DataFrame(synthetic_data)
    
    # Concatenate
    df_combined = pd.concat([df, aug_df], ignore_index=True)
    print(f"   > Total training samples after augmentation: {len(df_combined)}")
    return df_combined

df = load_and_augment_data()

# 2. Preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text) # Replace punctuation with space to keep words distinct
    return text.strip()

df['clean_review'] = df['review_text'].apply(clean_text)

# --- MODEL A: SENTIMENT ANALYSIS (TF-IDF + LR) ---
print("--- Training Model A: Sentiment Analysis ---")
X = df['clean_review']
y_sentiment = df['sentiment']

# Stratified Split to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(X, y_sentiment, test_size=0.1, random_state=42, stratify=y_sentiment)

# Robust Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=15000)),
    ('clf', LinearSVC(class_weight='balanced', random_state=42, C=1.0))
])

pipeline.fit(X_train, y_train)
print(f"   > Sentiment Accuracy: {pipeline.score(X_test, y_test):.4f}")
joblib.dump(pipeline, os.path.join(BASE_DIR, 'sentiment_model.pkl'))

# --- MODEL B: ISSUE IDENTIFICATION (Weak Supervision) ---
print("\n--- Training Model B: Issue Identification ---")

# Improved Weak Supervision Logic
def auto_label_issue(text):
    text = text.lower()
    
    # Priority Keywords (Logistics)
    if any(w in text for w in ['delivery', 'late', 'arrive', 'shipping', 'package', 'courier', 'track', 'ship', 'lost', 'delay']):
        return "Logistics"
    
    # Priority Keywords (Quality)
    if any(w in text for w in ['damaged', 'broken', 'defective', 'quality', 'working', 'stopped', 'useless', 'bad', 'terrible', 'worst', 'awful']):
        return "Quality"
        
    # Priority Keywords (Billing/Price)
    if any(w in text for w in ['price', 'refund', 'money', 'cost', 'charge', 'expensive', 'bill', 'worth']):
        return "Billing"
    
    return "General"

df['issue_type'] = df['clean_review'].apply(auto_label_issue)

# Train Classifier
y_issue = df['issue_type']
X_train_iss, X_test_iss, y_train_iss, y_test_iss = train_test_split(X, y_issue, test_size=0.1, random_state=42, stratify=y_issue)

issue_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=10000)),
    ('clf', LinearSVC(class_weight='balanced', random_state=42))
])

issue_pipeline.fit(X_train_iss, y_train_iss)
print("   > Issue Classification Report:")
print(classification_report(y_test_iss, issue_pipeline.predict(X_test_iss)))

joblib.dump(issue_pipeline, os.path.join(BASE_DIR, 'issue_model.pkl'))
print("\nSuccess! Advanced models trained and saved.")
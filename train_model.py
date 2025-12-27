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
df = pd.read_csv(os.path.join(BASE_DIR, 'Customer_Sentiment.csv'))

# --- 1.1 In-Memory Data Augmentation (Fix for missing vocabulary) ---
# The original dataset lacks strong sentiment words like 'hate', 'love', 'terrible'.
# We inject them here so the model learns them, without modifying the user's CSV.
print("   > Augmenting data with synthetic examples for robustness...")
augmented_data = [
    # Negative
    {"review_text": "I hate this product, it is terrible.", "sentiment": "negative"},
    {"review_text": "This is the worst experience of my life.", "sentiment": "negative"},
    {"review_text": "Absolutely awful service, never buying again.", "sentiment": "negative"},
    {"review_text": "I am very angry with the quality.", "sentiment": "negative"},
    {"review_text": "Disgusting behavior from support.", "sentiment": "negative"},
    {"review_text": "The item is bad and useless.", "sentiment": "negative"},
    {"review_text": "Horrible shipping time, took forever.", "sentiment": "negative"},
    # Positive
    {"review_text": "I love this product, it is the best!", "sentiment": "positive"},
    {"review_text": "Amazing quality and super fast shipping.", "sentiment": "positive"},
    {"review_text": "Excellent customer service, very helpful.", "sentiment": "positive"},
    {"review_text": "Superb experience, highly recommended.", "sentiment": "positive"},
    {"review_text": "Best purchase I have ever made.", "sentiment": "positive"},
    {"review_text": "I am so happy with my order.", "sentiment": "positive"},
    {"review_text": "Fantastic value for money.", "sentiment": "positive"},
    # Neutral
    {"review_text": "It is okay, assumes it does the job.", "sentiment": "neutral"},
    {"review_text": "Nothing special, just average.", "sentiment": "neutral"},
    {"review_text": "Received the item today.", "sentiment": "neutral"},
] * 100 # Repeat 100 times to ensure these words are statistically significant and picked up by TF-IDF

aug_df = pd.DataFrame(augmented_data)
# Concatenate and fill missing columns with NaN (they aren't used for training anyway)
df = pd.concat([df, aug_df], ignore_index=True)
print(f"   > Added {len(aug_df)} synthetic rows (upsampled) for better coverage.")

# 2. Preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

df['clean_review'] = df['review_text'].apply(clean_text)

# --- MODEL A: SENTIMENT ANALYSIS (TF-IDF + LR) ---
print("--- Training Model A: Sentiment Analysis ---")
X = df['clean_review']
y_sentiment = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y_sentiment, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LinearSVC(class_weight='balanced', random_state=42))
])

# Hyperparameter Tuning
param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__max_features': [5000, 10000],
    'clf__C': [0.1, 1, 10]
}

print("   > Tuning hyperparameters (this may take a moment)...")
grid_search = GridSearchCV(pipeline, param_grid, cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"   > Best Props: {grid_search.best_params_}")
print("Sentiment Accuracy:", best_model.score(X_test, y_test))
joblib.dump(best_model, os.path.join(BASE_DIR, 'sentiment_model.pkl'))

# --- MODEL B: ISSUE IDENTIFICATION (Weak Supervision) ---
print("\n--- Training Model B: Issue Identification ---")

# Step 1: Create "Weak Labels" using rules (to train the model)
def auto_label_issue(text):
    if any(w in text for w in ['late', 'delivery', 'wait', 'shipping', 'courier', 'arrive', 'fast']):
        return "Logistics"
    elif any(w in text for w in ['broken', 'quality', 'stopped', 'working', 'damaged', 'defective', 'worst']):
        return "Quality"
    elif any(w in text for w in ['price', 'money', 'refund', 'expensive', 'worth', 'cost', 'bill', 'charge']):
        return "Billing"
    else:
        return "General"

df['issue_type'] = df['clean_review'].apply(auto_label_issue)

# Step 2: Train a Classifier on these labels
y_issue = df['issue_type']
X_train_issue, X_test_issue, y_train_issue, y_test_issue = train_test_split(X, y_issue, test_size=0.2, random_state=42)

# Use similar robust pipeline for Issues
issue_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))),
    ('clf', LinearSVC(class_weight='balanced', random_state=42))
])

issue_pipeline.fit(X_train_issue, y_train_issue)
print("Issue Classification Report:\n", classification_report(y_test_issue, issue_pipeline.predict(X_test_issue)))
joblib.dump(issue_pipeline, os.path.join(BASE_DIR, 'issue_model.pkl'))

print("\nSuccess! Both models saved.")
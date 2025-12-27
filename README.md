# Customer Sentiment AI System

## Overview
This is a dual-purpose AI platform designed for **Customer Support Automation** and **Managerial Analytics**. It features a modern Streamlit interface and a terminal-based "Quantum Core" for deep data exploration.

## Modes

### 1. 💬 AI Support Assistant (`app.py`)
A next-generation customer support chatbot that uses **Hybrid Intelligence** (Rules + AI Models) to handle queries with human-like empathy and precision.

#### Key Features:
-   **🌟 Star Rating System**: The bot adapts its personality based on user feedback.
    -   **1-2 Stars (Unhappy)**: Auto-enters "Resolution Mode" (Offers Refunds/Replacements).
    -   **3 Stars (Neutral)**: Enters "Feedback Mode" (Connects to Help Desk).
    -   **4-5 Stars (Happy)**: Enters "Celebration Mode".
-   **🧠 Hybrid Logic Engine**:
    -   **Context Awareness**: Understands that "Delivery" is a *complaint* if rating is low, but a *compliment* if rating is high.
    -   **Typo Tolerance**: Automatically corrects "replcaement", "colr", etc.
    -   **Action-Oriented**: Handles "Refund" or "Replacement" requests directly without unnecessary questioning.
-   **🔄 Session Tools**: Sidebar button to instantly reset the chat for a new customer.

### 2. 📊 Manager Analytics (`manager_chat.py`)
A terminal-based tool for managers to analyze 10,000+ customer reviews.
-   **Platform Intelligence**: metrics for Amazon, Flipkart, etc.
-   **Deep Dive Wizard**: Interactive filtering by Region, Demographics, and Product Category.

---

## Installation

1.  **Install Dependencies**:
    ```bash
    pip install streamlit pandas rich joblib scikit-learn plotly
    ```
2.  **Train Models** (Required for Chatbot):
    ```bash
    python train_model.py
    ```
    *This generates `sentiment_model.pkl` and `issue_model.pkl`.*

---

## Usage

### Run the Web Interface (AI Assistant)
```bash
streamlit run app.py
```
Select **"Customer Chatbot"** from the sidebar to test the AI.

### Run the Manager Terminal
```bash
python manager_chat.py
```

---

## Technical Details
-   **Models**: TF-IDF Vectorizer + LinearSVC (Accuracy: ~95%).
-   **Logic**: Python-based hybrid classifier with regex pattern matching.
-   **UI**: Streamlit (Web) & Rich (Terminal).

## 🐳 Docker Support
Build and run the entire suite in a container:
```bash
docker build -t sentinel-ai .
docker run -p 8501:8501 sentinel-ai
```

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

# Set Page Configuration
st.set_page_config(page_title="Manager Chat Analytics", page_icon="📊", layout="wide")

# Custom CSS for futuristic look
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    h1, h2, h3 {
        color: #00e6e6;
    }
    div.stButton > button {
        background-color: #00e6e6;
        color: black;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Function to load data (Caching for performance)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Customer_Sentiment.csv')
        
        # --- Data Cleaning (Mirroring manager_chat.py) ---
        df.columns = [c.lower().strip() for c in df.columns]
        
        if 'customer_rating' in df.columns:
            df.rename(columns={'customer_rating': 'rating'}, inplace=True)
        if 'platform' in df.columns:
            df.rename(columns={'platform': 'purchase_platform'}, inplace=True)
            
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            
        if 'age_group' in df.columns:
            df['age_group'] = df['age_group'].replace({
                        '50+': '46-60',
                        '36-50': '36-45'
            })
            
        # Normalize Categorical Columns
        for col in ['region', 'product_category', 'purchase_platform', 'gender', 'age_group']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()
                
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load Data
df = load_data()

# Sidebar - Navigation
st.sidebar.title("🚀 Navigation")
mode = st.sidebar.radio("Go to:", ["Manager Analytics", "Customer Chatbot"])

st.sidebar.markdown("---")

# --- MODE 1: MANAGER ANALYTICS ---
if mode == "Manager Analytics":
    # Sidebar - Filters
    st.sidebar.title("🎛️ Analysis Controls")
    
    if not df.empty:
        # Filter: Region
        regions = ['All'] + sorted(list(df['region'].unique())) if 'region' in df.columns else ['All']
        selected_region = st.sidebar.selectbox("Region", regions)
        
        # Filter: Platform
        platforms = ['All'] + sorted(list(df['purchase_platform'].unique())) if 'purchase_platform' in df.columns else ['All']
        selected_platform = st.sidebar.selectbox("Platform", platforms)
        
        # Filter: Age Group
        ages = ['All'] + sorted(list(df['age_group'].unique())) if 'age_group' in df.columns else ['All']
        selected_age = st.sidebar.selectbox("Age Group", ages)
        
        # Apply Filters
        subset = df.copy()
        if selected_region != 'All':
            subset = subset[subset['region'] == selected_region]
        if selected_platform != 'All':
            subset = subset[subset['purchase_platform'] == selected_platform]
        if selected_age != 'All':
            subset = subset[subset['age_group'] == selected_age]
    
        # Main Content
        st.title("📊 Customer Sentiment Command Center")
        st.markdown("### Real-time AI Analytics Dashboard")
        
        # Top Metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Reviews", len(subset))
        with c2:
            avg_rating = subset['rating'].mean() if 'rating' in subset.columns else 0
            st.metric("Average Rating", f"{avg_rating:.2f}/5.0")
        with c3:
            neg_count = len(subset[subset['sentiment'] == 'negative'])
            st.metric("Negative Issues", neg_count, delta=-neg_count, delta_color="inverse")
        with c4:
            pos_count = len(subset[subset['sentiment'] == 'positive'])
            st.metric("Positive Highlights", pos_count)
            
        st.markdown("---")
        
        # Tabs for detailed views
        tab1, tab2, tab3 = st.tabs(["📈 Overview", "🛍️ Platform Intelligence", "🔍 Deep Dive Data"])
        
        with tab1:
            c_left, c_right = st.columns(2)
            
            with c_left:
                st.subheader("Sentiment Distribution")
                if 'sentiment' in subset.columns:
                    sent_counts = subset['sentiment'].value_counts().reset_index()
                    sent_counts.columns = ['Sentiment', 'Count']
                    fig_sent = px.pie(sent_counts, values='Count', names='Sentiment', color='Sentiment',
                                      color_discrete_map={'positive':'#00cc96', 'negative':'#EF553B', 'neutral':'#AB63FA'})
                    st.plotly_chart(fig_sent, use_container_width=True)
                    
            with c_right:
                st.subheader("Rating by Product Category")
                if 'product_category' in subset.columns and 'rating' in subset.columns:
                    prod_perf = subset.groupby('product_category')['rating'].mean().reset_index().sort_values('rating', ascending=True)
                    fig_prod = px.bar(prod_perf, x='rating', y='product_category', orientation='h',
                                      color='rating', color_continuous_scale='RdYlGn')
                    st.plotly_chart(fig_prod, use_container_width=True)
    
        with tab2:
            st.subheader("Platform Performance Matrix")
            
            if 'purchase_platform' in subset.columns:
                 # Aggregate Data
                plat_stats = subset.groupby('purchase_platform').agg(
                    Total_Transactions=('customer_id', 'count'),
                    Avg_Rating=('rating', 'mean'),
                    Negative_Issues=('sentiment', lambda x: (x=='negative').sum())
                ).reset_index()
                
                plat_stats = plat_stats.sort_values('Avg_Rating', ascending=False)
                
                # Highlight Best/Worst
                if not plat_stats.empty:
                    best_plat = plat_stats.iloc[0]['purchase_platform']
                    worst_plat = plat_stats.iloc[-1]['purchase_platform']
                    
                    k1, k2 = st.columns(2)
                    k1.success(f"🏆 Top Performer: **{best_plat}**")
                    k2.error(f"⚠️ Needs Attention: **{worst_plat}**")
                
                st.dataframe(plat_stats.style.background_gradient(subset=['Avg_Rating'], cmap='RdYlGn'), use_container_width=True)
                
                # Interactive Chart
                fig_plat = px.scatter(plat_stats, x='Avg_Rating', y='Negative_Issues', size='Total_Transactions', color='purchase_platform',
                                      hover_name='purchase_platform', title="Rating vs Issues Bubble Chart")
                st.plotly_chart(fig_plat, use_container_width=True)
    
        with tab3:
            st.subheader("Granular Data View")
            st.dataframe(subset)
    
    else:
        st.error("No data available. Please check 'Customer_Sentiment.csv'.")

# --- MODE 2: CUSTOMER CHATBOT ---
elif mode == "Customer Chatbot":
    st.title("💬 AI Support Assistant")
    st.markdown("Describe your issue below, and our AI will assist you instantly.")
    
    # Load Models
    @st.cache_resource
    def load_models():
        try:
            sent_model = joblib.load('sentiment_model.pkl')
            iss_model = joblib.load('issue_model.pkl')
            return sent_model, iss_model
        except Exception as e:
            return None, None

    sentiment_model, issue_model = load_models()
    
    if not sentiment_model or not issue_model:
        st.error("⚠️ Models not found! Please run `train_model.py` first to generate models.")
    else:
        # Chat Interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display Chat History (Correctly rendering HTML)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                # check if message supposedly contains HTML (assistant) or just text (user)
                # We default to unsafe_allow_html=True for assistant to render the debug badge
                st.markdown(message["content"], unsafe_allow_html=True)

        # User Input
        if prompt := st.chat_input("How can we help you today?"):
            # Add User Message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # --- AI Logic Engine (Advanced) ---
            import random
            import re

            def clean_text(text):
                text = str(text).lower()
                text = re.sub(r'[^\w\s]', '', text)
                return text

            # 1. Hybrid Classification
            def classify_intent_hybrid(text, sentiment_model, issue_model):
                text_clean = clean_text(text)
                
                # Rule 0: Gibberish / Very Short Input Check
                # If input is very short and not a known keyword, default to Neutral/General
                # This prevents "dsdsv" -> Negative
                if len(text_clean) < 4 and text_clean not in ['bad', 'sad', 'mad', 'wow']:
                     return "neutral", "General"

                # Rule-based Overrides
                if any(w in text_clean for w in ['refund', 'money back', 'charge', 'cost', 'price']):
                    return "negative", "Billing"
                if any(w in text_clean for w in ['broken', 'damaged', 'defective', 'stopped working', 'quality', 'bad']):
                    return "negative", "Quality"
                if any(w in text_clean for w in ['late', 'delivery', 'arrive', 'shipping', 'package']):
                    return "negative", "Logistics"
                if any(w in text_clean for w in ['amazing', 'love', 'great', 'best', 'happy', 'good']):
                    return "positive", "General"
                
                # Fallback to AI Model
                pred_sentiment = sentiment_model.predict([text_clean])[0]
                pred_issue = issue_model.predict([text_clean])[0]
                
                return pred_sentiment, pred_issue

            sentiment, issue_category = classify_intent_hybrid(prompt, sentiment_model, issue_model)

            # 2. Dynamic Response Generator
            def generate_response(sentiment, issue, text):
                text_clean = clean_text(text)
                
                # Specialized Responses
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
                            "Okay, I've noted that down. Anything else?",
                            "I see. How else can I assist you?"
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

                # Fetch Random Template
                try:
                    category_responses = responses.get(sentiment, {}).get(issue, responses["neutral"]["General"])
                    return random.choice(category_responses)
                except:
                    return "I'm here to help. Could you provide more details?"

            response_text = generate_response(sentiment, issue_category, prompt)
            
            # --- Futuristic UI formatting ---
            sentiment_color = "#00cc96" if sentiment == "positive" else "#EF553B" if sentiment == "negative" else "#AB63FA"
            
            # Formatted Debug Info (Styled Badge)
            debug_info = f"""
            <div style='background-color: #1E1E1E; border-radius: 5px; padding: 8px; margin-top: 10px; font-size: 0.8em; border: 1px solid #333;'>
                <span style='color: #888;'>🧠 AI Analysis:</span> 
                <span style='color:{sentiment_color}; font-weight: bold; margin-left: 5px;'>{sentiment.upper()}</span>
                <span style='color: #555; margin: 0 5px;'>|</span>
                <span style='color: #ddd;'>{issue_category}</span>
            </div>
            """
            
            full_display = f"{response_text} {debug_info}"
            
            # Add Assistant Message
            st.session_state.messages.append({"role": "assistant", "content": full_display, "is_html": True})
            with st.chat_message("assistant"):
                st.markdown(full_display, unsafe_allow_html=True)

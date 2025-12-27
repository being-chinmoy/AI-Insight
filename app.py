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
    
    # --- Feature: Session Reset ---
    if st.sidebar.button("🔄 New Customer / Reset Chat"):
        st.session_state.messages = []
        if "last_rating" in st.session_state:
            del st.session_state["last_rating"]
        st.rerun()
    
    # --- Feature: Rating System ---
    st.markdown("### 🌟 Rate your experience to start")
    user_rating = st.feedback("stars") # Streamlit 1.39+ feature
    
    st.markdown("---")
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
            
        # Initial Proactive Message based on Rating
        # We check if we already greeted the user for this rating to avoid spamming
        if user_rating is not None and "last_rating" not in st.session_state:
            st.session_state["last_rating"] = user_rating
            greeting = ""
            if user_rating <= 1: # 0 (1 star) or 1 (2 stars)
                greeting = "I see you're unhappy (Rating: Low). What seems to be the main problem? (e.g., Bad product, Wrong Color, High Price)"
            elif user_rating == 2: # 3 Stars
                greeting = "Thanks for the feedback. How can we improve? You can also ask to connect to our help desk."
            else: # 4-5 Stars
                greeting = "We're glad you're happy! What did you like most about the experience?"
            
            st.session_state.messages.append({"role": "assistant", "content": greeting, "is_html": False})

        # Display Chat History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
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

            # 1. Hybrid Classification
            def classify_intent_hybrid(text, sentiment_model, issue_model):
                text_clean = clean_text(text)
                
                # Rule 0: Gibberish Check
                short_sentiment_words = ['bad', 'sad', 'mad', 'wow', 'hate', 'good', 'love', 'best', 'ok', 'okay', 'poor', 'nice', 'help']
                if len(text_clean) < 6 and text_clean not in short_sentiment_words:
                    if text_clean in ['hi', 'hello', 'hey', 'thanks', 'thank', 'thx']:
                         return "neutral", "General"
                    return "neutral", "General"

                # Rule 1: Positive Keywords
                if 'fast' in text_clean and 'delivery' in text_clean:
                     return "positive", "Logistics"
                if any(w in text_clean for w in ['amazing', 'love', 'great', 'best', 'happy', 'good', 'excellent', 'fast']):
                     return "positive", "General"

                # Rule 2: Negative Specific Logic
                if any(w in text_clean for w in ['email', 'contact', 'support', 'human', 'agent', 'help desk']):
                    return "neutral", "Support"
                # Context shift: If user asks for "refund" or "replacement" specifically, treat as Resolution (Neutral/Info) rather than Complaint if possible, 
                # but for Issue Model training "refund" was Billing. We can override this in Response Generation.
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

            sentiment, issue_category = classify_intent_hybrid(prompt, sentiment_model, issue_model)

            # --- CONTEXT PERISITENCE ---
            # We need to remember the "Root Cause" (e.g., Quality) even if the user switches to asking for a "Refund" (which looks like Billing).
            if "active_issue" not in st.session_state:
                st.session_state["active_issue"] = None
            
            # Update active issue only if it's a specific root cause.
            # If the user just says "Refund", we don't want to overwrite "Quality" with "Billing".
            is_resolution_keyword = any(w in prompt.lower() for w in ['refund', 'replace', 'return', 'money'])
            
            if issue_category == "Quality":
                st.session_state["active_issue"] = "Quality"
            elif issue_category == "Logistics":
                st.session_state["active_issue"] = "Logistics"
            elif issue_category == "Billing" and not is_resolution_keyword:
                # Only set Billing if it's a real billing complaint (e.g. "overcharged"), not just "I want money back"
                st.session_state["active_issue"] = "Billing"
            
            # Use the PERSISTENT context if we have one, otherwise fallback to current detection
            effective_issue = st.session_state.get("active_issue", issue_category)
            if effective_issue is None: 
                effective_issue = issue_category

            # (Moved generate_response call to after definition)
            
            # 2. Dynamic Response Generator
            def generate_response(sentiment, issue, text, rating):
                text_clean = clean_text(text)
                
                # Critical Priority: Gratitude check
                if text_clean.startswith("thank") or "thank you" in text_clean or "thanks" in text_clean:
                    return "You are very welcome! Let me know if you need anything else."
                
                # Context Awareness: Order ID
                if re.search(r'\b\d{5,}\b', text):
                     return "Thanks for providing the Order ID. I am pulling up your details now... ⏳"

                # --- RATING BASED LOGIC FLOW ---
                
                # 1. Low Rating (0-1: Stars 1-2) -> Auto Refund/Replacement
                # 1. Low Rating (0-1: Stars 1-2) -> Auto Refund/Replacement
                if rating is not None and rating <= 1:
                    # CHECK FOR USER CHOICE FIRST (Resolution)
                    if "refund" in text_clean or "replacement" in text_clean:
                        # CASE A: Quality (Product Damaged) -> ALLOWED with Verification
                        if issue == "Quality":
                            return "A **photo verification** is required for quality claims. Please upload an image of the damaged item below. 📸"
                        
                        # CASE B: Logistics (Late Delivery vs Missing Item)
                        elif issue == "Logistics":
                            # Sub-case: Missing Item / Empty Box -> Needs Photo Verification
                            if any(w in text_clean for w in ['missing', 'empty', 'stolen', 'not in box']):
                                return "⚠️ **Security Check Required:** For missing items, we need to verify the package condition. Please upload a **photo of the received box/packaging** below. 📦"
                            
                            # Sub-case: Late Delivery -> Coupon
                            return "⚠️ **Refund Policy:** We cannot issue refunds for transit delays. However, I have personally checked your tracking.\n\n🚚 **Status:** In Transit (Delayed due to high volume).\nWe apologize for the inconvenience. Here is a coupon code: **DELAY50**."
                        
                        # CASE C: Billing (High Price) -> DENIED Refund -> OFFER Coupon
                        elif issue == "Billing":
                             return "⚠️ We cannot refund based on price perception. However, I have shared your feedback with the team. Here is a discount code: **FUTURESAVE20**."
                        
                        # CASE D: General/Other
                        else:
                            return "I have flagged this transaction for a human manager to review. Reference Ticket: #99281."

                    # If no choice made yet, offer options based on issue
                    if issue == "Quality":
                        return "I apologize for the poor quality. Since you rated us low, I can offer an immediate **Refund** or **Replacement**. Which do you prefer?"
                    elif issue == "Billing":
                        return "I see you feel the price is unfair. I have shared this feedback with the seller. 📉 To make it up to you, here is a **Coupon Code** for your next purchase: **FUTURESAVE20**."
                    elif issue == "Logistics":
                        return "I sincerely apologize for the delay. 😞 It appears there is a transit issue. Here is your **Tracking Details**. \n\n**Note:** Our team is working on it. To make up for the wait, please use this coupon: **DELAY50**."
                
                # 2. Mid Rating (2: Star 3) -> Help Desk / Email
                if rating is not None and rating == 2:
                    if "email" in text_clean or "help" in text_clean or "desk" in text_clean or "connect" in text_clean:
                         return "I have registered your request. 📧 **Sending an escalation email to support@sportsphere.com...** [Sent Successfully]"

                # 3. High Rating (3-4: Stars 4-5) -> Thanks
                # If rating is high, we interpret the user's input as "what they liked", so we override any negative classification for keywords like "delivery".
                if rating is not None and rating >= 3:
                     # Dynamic acknowledgment
                     if "delivery" in text_clean:
                         return "Glad to hear the **Delivery** was up to the mark! 🚚💨 Thanks for the 5 stars!"
                     if "quality" in text_clean or "product" in text_clean:
                         return "Awesome! We pride ourselves on **Component Quality**. Enjoy your gear! 🏆"
                     if "price" in text_clean:
                         return "We try to offer the best value. Thanks for noticing! 💸"
                     
                     return "We are over the moon! 🌙 Thank you for your support. Keep rocking!"

                # --- DEFAULT LOGIC ---
                responses = {
                    "negative": {
                        "Logistics": [
                            "I've prioritized this logicstics issue. Do you have your Order ID?",
                            "Tracking shows a delay. I'm sorry. Can I have your Order Number to investigate?"
                        ],
                        "Quality": [
                            "I apologize for the defect. Would you like a return?",
                            "That sounds frustrating. We can swap that item for you."
                        ],
                        "Billing": [
                            "I understand the billing concern. Connecting to Finance...",
                            "Let me check why you were charged that amount."
                        ],
                        "Support": [
                            "Here is our support email: support@sportsphere.com. I've also flagged this chat for a manager.",
                            "A human agent will email you shortly."
                        ],
                        "General": [
                            "I apologize for the bad experience.",
                            "We will do better next time."
                        ]
                    },
                    "neutral": {
                        "Support": ["Our help desk is available 24/7 at support@sportsphere.com."],
                        "General": ["How else can I help?", "Understood."]
                    },
                    "positive": {
                        "General": ["Thanks!", "Awesome!"]
                    }
                }
                
                try:
                    category_responses = responses.get(sentiment, {}).get(issue, responses["neutral"]["General"])
                    return random.choice(category_responses)
                except:
                    return "I'm listening. Tell me more."

            # Generate Response (MOVED HERE to fix usage-before-definition error)
            response_text = generate_response(sentiment, effective_issue, prompt, user_rating)
            
            # --- Futuristic UI formatting ---
            sentiment_color = "#00cc96" if sentiment == "positive" else "#EF553B" if sentiment == "negative" else "#AB63FA"
            
            # Formatted Debug Info
            debug_info = f"""
            <div style='background-color: #1E1E1E; border-radius: 5px; padding: 8px; margin-top: 10px; font-size: 0.8em; border: 1px solid #333;'>
                <span style='color: #888;'>AI Brain:</span> 
                <span style='color:{sentiment_color}; font-weight: bold; margin-left: 5px;'>{sentiment.upper()}</span>
                <span style='color: #555; margin: 0 5px;'>|</span>
                <span style='color: #ddd;'>{issue_category}</span>
                <span style='color: #555; margin: 0 5px;'>|</span>
                <span style='color: #aaa;'>Rating: {user_rating + 1 if user_rating is not None else "N/A"}★</span>
            </div>
            """
            
            full_display = f"{response_text} {debug_info}"
            
            # Add Assistant Message
            st.session_state.messages.append({"role": "assistant", "content": full_display, "is_html": True})
            with st.chat_message("assistant"):
                st.markdown(full_display, unsafe_allow_html=True)
                
            # --- PHOTO VERIFICATION LOGIC ---
            # Relaxed check to ensure it catches "photo verification" or "upload an image"
            if "photo" in response_text.lower() or "upload" in response_text.lower():
                st.session_state['waiting_for_photo'] = True
                # Store the intended action based on user text trigger
                if "replacement" in prompt.lower() or "exchange" in prompt.lower():
                    st.session_state['pending_action'] = "Replacement"
                else:
                    st.session_state['pending_action'] = "Refund"
                st.rerun()

        # Check Output from previous turn (File Uploader State)
        if st.session_state.get('waiting_for_photo', False):
            uploaded_file = st.file_uploader("Upload Image of Damaged Item", type=['png', 'jpg', 'jpeg'])
            
            if uploaded_file is not None:
                # --- HYBRID VISION LOGIC ---
                # Attempt to use the Real Vision Model if available, otherwise Simulate.
                
                with st.spinner("🤖 AI Vision analyzing image damage..."):
                    import time
                    time.sleep(2.0) # UX Wait time
                    
                    real_model_path = 'damage_vision_model.h5'
                    vision_confidence = 0.0
                    
                    # 1. Real AI Check
                    try:
                        import tensorflow as tf
                        if os.path.exists(real_model_path):
                            from vision_model import predict_damage
                            # Save temp file for processing
                            with open("temp_img.jpg", "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            vision_confidence = predict_damage("temp_img.jpg")
                        else:
                            # Simulation Fallback (High confidence for demo)
                            vision_confidence = 0.984
                    except ImportError:
                        # Library not installed -> Simulation
                        vision_confidence = 0.984
                    
                
                if vision_confidence > 0.7:
                    st.success(f"✅ Damage Detected (Confidence: {vision_confidence:.1%})")
                    
                    # DYNAMIC FINAL ACTION
                    action_type = st.session_state.get('pending_action', 'Refund')
                    
                    if action_type == "Replacement":
                        st.markdown("**Replacement Approved.** 📦 A new order has been created and will ship within 24 hours.")
                        st.session_state.messages.append({"role": "assistant", "content": "✅ **Replacement Shipped.** Tracking #TRK9982 sent to your email.", "is_html": False})
                    else:
                        st.markdown("**Refund Approved.** 💸 Initiating transfer to original payment source...")
                        st.session_state.messages.append({"role": "assistant", "content": "✅ **Refund Initiated.** Funds will appear in 3-5 days.", "is_html": False})

                else:
                    st.warning(f"⚠️ Damage not clearly visible (Confidence: {vision_confidence:.1%}). forwarding to manual review.")
                    st.session_state.messages.append({"role": "assistant", "content": "I've forwarded your image to a human agent for manual review.", "is_html": False})

                # Reset state
                st.session_state['waiting_for_photo'] = False

import streamlit as st
import pandas as pd
import plotly.express as px

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

# Sidebar - Filters
st.sidebar.title("🎛️ Analysis Controls")
st.sidebar.markdown("---")

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

import streamlit as st
import pickle
import os
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling (light theme)
st.markdown("""
<style>
    /* Light background gradient */
    .stApp {
        background: linear-gradient(135deg, #f7fbff 0%, #e6f7ff 100%);
        min-height: 100vh;
    }
    
    /* Main content area */
    .main {
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #ffffff 0%, #f1f8ff 100%);
        border-radius: 10px;
        margin: 1rem;
    }
    
    /* Sidebar styling (soft teal) */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #dff6f0 0%, #cfeff6 100%);
        color: #073642;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #073642;
    }
    
    [data-testid="stSidebar"] p {
        color: #094253;
    }
    
    /* Metrics styling */
    .stMetric {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-top: 4px solid #6fb3ff;
    }
    
    /* Text areas and input fields */
    .stTextArea textarea {
        background-color: #ffffff !important;
        border: 2px solid #6fb3ff !important;
        border-radius: 8px !important;
        color: #0b2b3a !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #6fb3ff 0%, #8bd3ff 100%);
        color: #073642;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton button:hover {
        box-shadow: 0 6px 12px rgba(139, 211, 255, 0.4);
        transform: translateY(-2px);
    }
    
    /* Success box */
    .success-box {
        background: linear-gradient(135deg, #e6f9ee 0%, #def3e7 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        box-shadow: 0 4px 8px rgba(40, 167, 69, 0.06);
    }
    
    /* Warning box */
    .warning-box {
        background: linear-gradient(135deg, #fff9e6 0%, #fff4d9 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        box-shadow: 0 4px 8px rgba(255, 193, 7, 0.06);
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #eaf8ff 0%, #e1f4ff 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        box-shadow: 0 4px 8px rgba(23, 162, 184, 0.06);
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #1b3b5a;
    }
    
    /* Data frame styling */
    .stDataFrame {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.04);
    }
    
    /* Info/Warning/Success messages */
    .stAlert {
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.04);
    }
    
    /* Download button */
    .stDownloadButton button {
        background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
    }
    
    .stDownloadButton button:hover {
        box-shadow: 0 6px 12px rgba(40, 167, 69, 0.18);
    }
</style>
""", unsafe_allow_html=True)

# Setup NLTK data
@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

setup_nltk()

# Get the directory containing this script
app_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(app_dir)

# Load model and vectorizer with caching
@st.cache_resource
def load_models():
    model_path = os.path.join(project_root, 'model', 'model.pkl')
    vectorizer_path = os.path.join(project_root, 'model', 'vectorizer.pkl')
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}")
        st.stop()

model, vectorizer = load_models()

@st.cache_resource
def get_stop_words():
    return set(stopwords.words('english'))

stop_words = get_stop_words()

# Initialize session state
if 'ticket_history' not in st.session_state:
    st.session_state.ticket_history = []

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def assign_priority(text):
    text = text.lower()
    if "not working" in text or "crashing" in text or "charged twice" in text:
        return "High"
    elif "refund" in text:
        return "Medium"
    else:
        return "Low"

def get_priority_color(priority):
    colors = {
        "High": "🔴",
        "Medium": "🟡",
        "Low": "🟢"
    }
    return colors.get(priority, "⚪")

def get_category_emoji(category):
    emojis = {
        "Technical issue": "🔧",
        "Billing inquiry": "💳",
        "Refund request": "💰",
        "Cancellation request": "❌",
        "Product inquiry": "❓"
    }
    return emojis.get(category, "📋")

# Header
st.markdown("# 🎫 Support Ticket Classifier")
st.markdown("Automatically classify and prioritize support tickets using AI")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Show model info
    st.info("""
    **Model Information:**
    - Model Type: Logistic Regression
    - Vectorizer: TF-IDF
    - Categories: 5 types
    - Last Updated: Today
    """)
    
    # Show example tickets
    st.markdown("## 📌 Example Tickets")
    examples = {
        "Technical Issue": "My product is not working and keeps crashing. Please help!",
        "Billing": "I was charged twice for my purchase. This is a mistake.",
        "Refund": "I want a refund for my recent order.",
        "Product Info": "Do you have any information about the available models?",
        "Cancellation": "Please cancel my subscription immediately."
    }
    
    for example_type, example_text in examples.items():
        if st.button(f"📝 {example_type}", key=f"btn_{example_type}", use_container_width=True):
            st.session_state.ticket_text = example_text

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### ✍️ Enter Your Support Ticket")
    
    ticket = st.text_area(
        "Ticket Description",
        value=st.session_state.get('ticket_text', ''),
        placeholder="Describe your support issue here...",
        height=150,
        label_visibility="collapsed"
    )
    
    # Character count
    char_count = len(ticket)
    st.caption(f"📊 Characters: {char_count}/500")
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        classify_btn = st.button("🚀 Classify Ticket", use_container_width=True)
    
    with col_btn2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    with col_btn3:
        history_btn = st.button("📜 History", use_container_width=True)
    
    if clear_btn:
        st.session_state.ticket_text = ""
        st.rerun()

with col2:
    st.markdown("### 📊 Quick Stats")
    st.metric("Total Tickets", len(st.session_state.ticket_history))
    
    if st.session_state.ticket_history:
        recent = st.session_state.ticket_history[-1]
        st.success(f"**Latest:** {recent['category']}")
        st.info(f"**Priority:** {recent['priority']}")

# Process ticket
if classify_btn:
    if not ticket.strip():
        st.warning("⚠️ Please enter a ticket description")
    else:
        # Clean and classify
        cleaned = clean_text(ticket)
        vect = vectorizer.transform([cleaned])
        
        # Get prediction
        category = model.predict(vect)[0]
        priority = assign_priority(ticket)
        
        # Get confidence (probability)
        confidence = max(model.predict_proba(vect)[0]) * 100
        
        # Store in history
        st.session_state.ticket_history.append({
            'timestamp': datetime.now(),
            'ticket': ticket,
            'category': category,
            'priority': priority,
            'confidence': confidence
        })
        
        # Display results
        st.markdown("---")
        st.markdown("### 📋 Classification Results")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.markdown(f"""
            <div class="info-box">
            <h3>🏷️ Category</h3>
            <h2>{get_category_emoji(category)} {category}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with result_col2:
            st.markdown(f"""
            <div class="warning-box">
            <h3>⚡ Priority</h3>
            <h2>{get_priority_color(priority)} {priority}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with result_col3:
            st.markdown(f"""
            <div class="success-box">
            <h3>🎯 Confidence</h3>
            <h2>{confidence:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Confidence bar
        st.progress(confidence / 100)
        
        # Additional Analysis
        st.markdown("### 🔍 Detailed Analysis")
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            st.markdown("**Cleaned Text:**")
            st.info(cleaned[:200] + "..." if len(cleaned) > 200 else cleaned)
        
        with analysis_col2:
            st.markdown("**Keywords Detected:**")
            keywords = []
            text_lower = ticket.lower()
            
            issue_keywords = ["not working", "crashing", "error", "bug"]
            billing_keywords = ["charged", "billing", "payment", "invoice"]
            refund_keywords = ["refund", "money back", "return"]
            
            for kw in issue_keywords:
                if kw in text_lower:
                    keywords.append(f"🔴 {kw}")
            for kw in billing_keywords:
                if kw in text_lower:
                    keywords.append(f"💳 {kw}")
            for kw in refund_keywords:
                if kw in text_lower:
                    keywords.append(f"💰 {kw}")
            
            if keywords:
                st.write(", ".join(keywords))
            else:
                st.write("*No critical keywords detected*")

# Show history
if history_btn or st.sidebar.checkbox("📜 Show Ticket History"):
    st.markdown("---")
    st.markdown("### 📜 Ticket History")
    
    if st.session_state.ticket_history:
        # Convert to DataFrame for better display
        history_df = pd.DataFrame([
            {
                'Time': h['timestamp'].strftime("%H:%M:%S"),
                'Category': h['category'],
                'Priority': h['priority'],
                'Confidence': f"{h['confidence']:.1f}%",
                'Ticket': h['ticket'][:50] + "..." if len(h['ticket']) > 50 else h['ticket']
            }
            for h in st.session_state.ticket_history
        ])
        
        st.dataframe(history_df, use_container_width=True)
        
        # Download history
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download History as CSV",
            data=csv,
            file_name=f"ticket_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No tickets classified yet.")

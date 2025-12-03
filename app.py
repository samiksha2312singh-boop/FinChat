"""
FinChat AI - Multi-Agent Financial Analysis System
Powered by Fireworks Llama 3.1 70B, LangChain, and RAG
"""

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import json
from datetime import datetime
import re

# Custom imports
from utils.agents import FinancialAgentOrchestrator
from utils.rag import ingest_filing_to_rag, clear_filing_data
from utils.tools import get_stock_metrics_tool

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="FinChat AI - Multi-Agent Financial Analysis",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE
# ============================================================================

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'uploaded_filings' not in st.session_state:
    st.session_state.uploaded_filings = []

if 'portfolio_config' not in st.session_state:
    st.session_state.portfolio_config = {
        'risk_tolerance': 'Moderate',
        'investment_horizon': '3-5 years',
        'portfolio_allocation': 10.0
    }

if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = 'AAPL'

# ============================================================================
# STYLING
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B35, #F7931E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# POPULAR TICKERS
# ============================================================================

POPULAR_TICKERS = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'META': 'Meta Platforms',
    'NVDA': 'NVIDIA Corporation',
    'TSLA': 'Tesla Inc.',
    'AMZN': 'Amazon.com Inc.',
    'JPM': 'JPMorgan Chase',
    'V': 'Visa Inc.',
    'JNJ': 'Johnson & Johnson',
    'WMT': 'Walmart',
    'PG': 'Procter & Gamble',
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_filing_sections(text: str) -> dict:
    """Extract key sections from SEC filing text"""
    sections = {}
    
    # Risk Factors
    risk_patterns = [
        r"RISK\s+FACTORS.*?(?=ITEM|UNREGISTERED|$)",
        r"ITEM\s+1A\..*?(?=ITEM\s+2|$)"
    ]
    for pattern in risk_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            sections['Risk Factors'] = match.group(0)[:15000]
            break
    
    # Management Discussion & Analysis
    mda_patterns = [
        r"MANAGEMENT'?S\s+DISCUSSION\s+AND\s+ANALYSIS.*?(?=QUANTITATIVE|ITEM\s+3|$)",
        r"MD&A.*?(?=ITEM\s+3|$)"
    ]
    for pattern in mda_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            sections['Management Discussion & Analysis'] = match.group(0)[:15000]
            break
    
    if not sections:
        sections['Full Text'] = text[:15000]
    
    return sections


def process_uploaded_file(uploaded_file, ticker: str):
    """Process uploaded SEC filing"""
    try:
        if uploaded_file.type == "text/plain":
            content = uploaded_file.read().decode('utf-8')
        elif uploaded_file.type == "application/pdf":
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text()
            except ImportError:
                return {'error': 'PyPDF2 not installed'}
        else:
            content = uploaded_file.read().decode('utf-8', errors='ignore')
        
        sections = extract_filing_sections(content)
        
        return {
            'ticker': ticker,
            'filename': uploaded_file.name,
            'upload_time': datetime.now().isoformat(),
            'sections': sections,
            'full_text': content[:50000]
        }
    except Exception as e:
        return {'error': str(e)}

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### 🔥 FinChat AI")
    st.markdown("*Multi-Agent Financial Analysis*")
    st.divider()
    
    # Ticker Selection
    st.header("⚙️ Configuration")
    ticker = st.selectbox(
        "📈 Select Stock",
        options=list(POPULAR_TICKERS.keys()),
        format_func=lambda x: f"{x} - {POPULAR_TICKERS[x]}",
        index=0
    )
    st.session_state.current_ticker = ticker
    
    st.divider()
    
    # SEC Filing Upload
    st.subheader("📄 Upload SEC Filing")
    st.caption("Upload 10-Q or 10-K for deeper analysis")
    
    uploaded_file = st.file_uploader(
        "Choose file",
        type=['txt', 'pdf', 'html'],
        help="Upload SEC filing document"
    )
    
    if uploaded_file:
        filing_ticker = st.text_input(
            "Ticker for this filing",
            value=ticker,
            help="Which company is this filing for?"
        ).upper()
        
        if st.button("🔄 Process Filing", type="primary"):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                filing_data = process_uploaded_file(uploaded_file, filing_ticker)
                
                if 'error' not in filing_data:
                    ingest_filing_to_rag(filing_data, filing_ticker)
                    st.session_state.uploaded_filings.append(filing_data)
                    st.success(f"✅ Processed for {filing_ticker}")
                else:
                    st.error(filing_data['error'])
    
    # Show uploaded filings
    if st.session_state.uploaded_filings:
        st.caption("**📁 Uploaded Filings:**")
        for filing in st.session_state.uploaded_filings:
            st.caption(f"✓ {filing['ticker']}: {filing['filename'][:20]}...")
    
    st.divider()
    
    # Portfolio Settings
    with st.expander("💼 Portfolio Settings"):
        risk_tolerance = st.select_slider(
            "Risk Tolerance",
            options=['Conservative', 'Moderate', 'Aggressive'],
            value='Moderate'
        )
        
        investment_horizon = st.selectbox(
            "Investment Horizon",
            ['Short-term (< 1 year)', '1-3 years', '3-5 years', '5+ years'],
            index=2
        )
        
        portfolio_allocation = st.slider(
            "Target Allocation (%)",
            min_value=1.0,
            max_value=50.0,
            value=10.0,
            step=0.5
        )
        
        st.session_state.portfolio_config = {
            'risk_tolerance': risk_tolerance,
            'investment_horizon': investment_horizon,
            'portfolio_allocation': portfolio_allocation
        }
    
    st.divider()
    st.caption("**📊 Session Stats:**")
    st.caption(f"Queries: {len(st.session_state.messages) // 2}")
    st.caption(f"Filings: {len(st.session_state.uploaded_filings)}")
    
    st.divider()
    st.caption("**🔥 Powered by:**")
    st.caption("• Llama 3.1 70B (Fireworks)")
    st.caption("• LangChain Multi-Agent")
    st.caption("• RAG with ChromaDB")

# ============================================================================
# MAIN INTERFACE
# ============================================================================

st.markdown('<h1 class="main-header">🔥 FinChat AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Multi-Agent Financial Analysis • Powered by Llama 3.1 & Fireworks AI</p>', 
           unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["💬 Chat Analysis", "📊 Quick Metrics", "ℹ️ About"])

# ============================================================================
# TAB 1: CHAT INTERFACE
# ============================================================================

with tab1:
    ticker = st.session_state.current_ticker
    company_name = POPULAR_TICKERS.get(ticker, ticker)
    
    # Header
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.metric("Company", f"{ticker} - {company_name}")
    with col2:
        filing_count = len([f for f in st.session_state.uploaded_filings if f['ticker'] == ticker])
        status = "✅ Loaded" if filing_count > 0 else "⚠️ Upload 10-Q"
        st.metric("SEC Filing", status)
    with col3:
        st.metric("Agents", "5 Active")
    
    st.divider()
    
    # Example queries
    with st.expander("💡 Example Questions"):
        st.markdown(f"""
        **Investment Decisions:**
        - Should I invest in {ticker}?
        - What are good entry and exit prices?
        
        **Risk Analysis:**
        - What are the main risks for {ticker}?
        - How risky is this investment?
        
        **Product Performance:**
        - Which products are performing best?
        
        **Competitive Analysis:**
        - How does {ticker} compare to peers?
        """)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                st.caption("🔥 Llama 3.1 70B")
    
    # Chat input
    if prompt := st.chat_input(f"Ask about {ticker}'s financial performance..."):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            try:
                orchestrator = FinancialAgentOrchestrator(
                    ticker=ticker,
                    portfolio_config=st.session_state.portfolio_config
                )
                
                answer = orchestrator.route_query(prompt)
                st.markdown(answer)
                st.caption("🔥 Llama 3.1 70B")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
                
            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.error(error_msg)

# ============================================================================
# TAB 2: QUICK METRICS
# ============================================================================

with tab2:
    ticker = st.session_state.current_ticker
    st.subheader(f"📊 Quick Metrics for {ticker}")
    
    try:
        metrics_json = get_stock_metrics_tool(ticker)
        metrics = json.loads(metrics_json)
        
        if 'error' not in metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Price</h3>
                    <h1>${metrics['price']}</h1>
                    <p>P/E: {metrics['pe_ratio']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Market Cap</h3>
                    <h1>${metrics['market_cap_b']}B</h1>
                    <p>{metrics['sector']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Revenue Growth</h3>
                    <h1>{metrics['revenue_growth_pct']}%</h1>
                    <p>YoY</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Profit Margin</h3>
                    <h1>{metrics['profit_margin_pct']}%</h1>
                    <p>TTM</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💰 Valuation")
                st.dataframe(pd.DataFrame({
                    'Metric': ['Price', 'P/E', 'Market Cap', 'Beta'],
                    'Value': [f"${metrics['price']}", f"{metrics['pe_ratio']}", 
                             f"${metrics['market_cap_b']}B", f"{metrics['beta']}"]
                }), hide_index=True)
            
            with col2:
                st.subheader("📈 Performance")
                st.dataframe(pd.DataFrame({
                    'Metric': ['Revenue Growth', 'Profit Margin', 'D/E Ratio'],
                    'Value': [f"{metrics['revenue_growth_pct']}%", 
                             f"{metrics['profit_margin_pct']}%", 
                             f"{metrics['debt_to_equity']}"]
                }), hide_index=True)
            
            st.subheader("📈 6-Month Price Chart")
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo")
            
            if not hist.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist.index, y=hist['Close'],
                    mode='lines', name='Price',
                    line=dict(color='#FF6B35', width=2)
                ))
                fig.update_layout(
                    yaxis_title='Price ($)',
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Unable to fetch data for {ticker}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

# ============================================================================
# TAB 3: ABOUT
# ============================================================================

with tab3:
    st.header("About FinChat AI")
    
    st.markdown("""
    ### 🎯 Multi-Agent Financial Analysis System
    
    **FinChat AI** uses specialized AI agents to provide comprehensive stock analysis.
    
    ### 🤖 Specialized Agents
    
    - **💼 Investment Advisor** - Buy/sell recommendations with price targets
    - **🛡️ Risk Analyst** - Risk assessment from metrics & filings
    - **📦 Product Analyst** - Business segment performance
    - **🏆 Peer Comparator** - Competitive positioning
    - **📊 General Analyst** - Flexible financial analysis
    
    ### 🔥 Technology
    
    - **LLM:** Llama 3.1 70B (Fireworks AI)
    - **Framework:** LangChain Multi-Agent
    - **Embeddings:** OpenAI text-embedding-3-small
    - **Vector DB:** ChromaDB
    - **Data:** Yahoo Finance + SEC EDGAR
    
    ### 👥 Team
    
    **IST.688.M001.FALL25**
    - Bhushan Jain
    - Samiksha Singh
    - Anjali Kalra
    - Shraddha Aher
    
    ### ⚖️ Disclaimer
    
    Educational purposes only. Not financial advice.
    """)

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>FinChat AI v2.0</strong> | Llama 3.1 70B (Fireworks) & LangChain</p>
    <p>IST.688.M001.FALL25 - Building Human-Centered AI Applications</p>
</div>
""", unsafe_allow_html=True)
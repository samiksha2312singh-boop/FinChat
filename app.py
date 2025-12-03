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
# ENHANCED STYLING - FIXED TEXT SPACING
# ============================================================================

st.markdown("""
<style>
    /* Main Header */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B35, #F7931E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* FIX TEXT SPACING ISSUES */
    .stMarkdown p {
        word-spacing: 0.2em;
        letter-spacing: 0.015em;
        line-height: 1.8;
        word-break: break-word;
        white-space: pre-wrap;
        font-size: 1rem;
    }
    
    .stMarkdown strong {
        letter-spacing: 0.03em;
        word-spacing: 0.15em;
        font-weight: 600;
    }
    
    /* Headers with better spacing */
    .stMarkdown h2 {
        margin-top: 2rem;
        margin-bottom: 1rem;
        letter-spacing: 0.02em;
        word-spacing: 0.1em;
        line-height: 1.4;
    }
    
    .stMarkdown h3 {
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        letter-spacing: 0.02em;
        word-spacing: 0.1em;
    }
    
    /* Lists with better spacing */
    .stMarkdown li {
        margin-bottom: 0.8rem;
        line-height: 1.7;
        word-spacing: 0.15em;
    }
    
    .stMarkdown ul, .stMarkdown ol {
        padding-left: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Code blocks */
    .stMarkdown code {
        padding: 0.2rem 0.4rem;
        background-color: rgba(240, 242, 246, 0.8);
        border-radius: 3px;
        font-family: 'Monaco', monospace;
    }
    
    /* Chat messages - better readability */
    .stChatMessage {
        padding: 1rem;
    }
    
    /* Ensure proper text wrapping */
    .element-container {
        word-break: break-word;
        overflow-wrap: break-word;
    }
    
    /* Fix for inline code/bold in verdicts */
    .stMarkdown p strong {
        display: inline;
        margin-right: 0.3em;
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
    
    # Financial Statements
    financial_patterns = [
        r"CONSOLIDATED\s+(?:BALANCE\s+SHEETS?|STATEMENTS?).*?(?=NOTES|ITEM|$)",
        r"FINANCIAL\s+STATEMENTS.*?(?=NOTES|$)"
    ]
    for pattern in financial_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            sections['Financial Statements'] = match.group(0)[:15000]
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
                return {'error': 'PyPDF2 not installed. Run: pip install pypdf2'}
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
                    st.info(f"📋 Extracted {len(filing_data['sections'])} sections")
                else:
                    st.error(filing_data['error'])
    
    # Show uploaded filings
    if st.session_state.uploaded_filings:
        st.divider()
        st.caption("**📁 Uploaded Filings:**")
        for filing in st.session_state.uploaded_filings:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(f"✓ {filing['ticker']}: {filing['filename'][:25]}...")
            with col2:
                if st.button("🗑️", key=f"del_{filing['ticker']}_{filing['upload_time']}", help="Delete"):
                    try:
                        clear_filing_data(filing['ticker'])
                        st.session_state.uploaded_filings.remove(filing)
                        st.rerun()
                    except:
                        pass
    
    st.divider()
    
    # Portfolio Settings
    with st.expander("💼 Portfolio Settings"):
        st.caption("Customize analysis for your profile")
        
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
        
        st.caption(f"🎯 Current: {risk_tolerance} | {investment_horizon} | {portfolio_allocation}%")
    
    st.divider()
    
    # Session Stats
    st.caption("**📊 Session Stats:**")
    st.caption(f"💬 Queries: {len(st.session_state.messages) // 2}")
    st.caption(f"📄 Filings: {len(st.session_state.uploaded_filings)}")
    
    st.divider()
    
    # Tech Stack
    st.caption("**🔥 Powered by:**")
    st.caption("• Llama 3.1 70B (Fireworks)")
    st.caption("• LangChain Multi-Agent")
    st.caption("• RAG with ChromaDB")
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

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
    
    # Company Header
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.metric("Company", f"{ticker} - {company_name}")
    with col2:
        filing_count = len([f for f in st.session_state.uploaded_filings if f['ticker'] == ticker])
        status_text = "✅ Loaded" if filing_count > 0 else "⚠️ Upload 10-Q"
        st.metric("SEC Filing", status_text)
    with col3:
        st.metric("Agents", "5 Active")
    
    st.divider()
    
    # Example queries
    with st.expander("💡 Example Questions - Click to Try"):
        example_col1, example_col2 = st.columns(2)
        
        with example_col1:
            st.markdown("**Investment Decisions:**")
            if st.button(f"Should I invest in {ticker}?", key="ex1"):
                st.session_state.example_query = f"Should I invest in {ticker}?"
                st.rerun()
            if st.button(f"What are good entry prices for {ticker}?", key="ex2"):
                st.session_state.example_query = f"What are good entry prices for {ticker}?"
                st.rerun()
        
        with example_col2:
            st.markdown("**Analysis:**")
            if st.button(f"What are the main risks for {ticker}?", key="ex3"):
                st.session_state.example_query = f"What are the main risks for {ticker}?"
                st.rerun()
            if st.button(f"Compare {ticker} to its competitors", key="ex4"):
                st.session_state.example_query = f"Compare {ticker} to its competitors"
                st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                st.caption("🔥 Llama 3.1 70B (Fireworks AI)")
    
    # Chat input - handle both manual input and example queries
    chat_input_value = st.session_state.pop('example_query', '')
    
    if prompt := (chat_input_value or st.chat_input(f"Ask about {ticker}'s financial performance, risks, or investment potential...")):
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            try:
                # Initialize orchestrator
                orchestrator = FinancialAgentOrchestrator(
                    ticker=ticker,
                    portfolio_config=st.session_state.portfolio_config
                )
                
                # Get multi-agent response
                answer = orchestrator.route_query(prompt)
                
                st.markdown(answer)
                st.caption("🔥 Llama 3.1 70B (Fireworks AI)")
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
                
            except Exception as e:
                error_msg = f"⚠️ **Error:** {str(e)}\n\nPlease check your API keys in `.streamlit/secrets.toml`"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# ============================================================================
# TAB 2: QUICK METRICS
# ============================================================================

with tab2:
    ticker = st.session_state.current_ticker
    
    st.subheader(f"📊 Quick Metrics for {ticker}")
    st.caption(f"Analyzing {POPULAR_TICKERS.get(ticker, ticker)}")
    
    try:
        metrics_json = get_stock_metrics_tool(ticker)
        metrics = json.loads(metrics_json)
        
        if 'error' not in metrics:
            # Display data source
            if 'data_source' in metrics:
                st.info(f"📊 {metrics['data_source']}")
            elif 'note' in metrics:
                st.info(f"📊 {metrics['note']}")
            
            # Top 4 metrics cards
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
                growth_color = '#00CC96' if metrics['revenue_growth_pct'] > 10 else '#FFA500' if metrics['revenue_growth_pct'] > 0 else '#EF553B'
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Revenue Growth</h3>
                    <h1 style="color: {growth_color};">{metrics['revenue_growth_pct']}%</h1>
                    <p>Year over Year</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                margin_color = '#00CC96' if metrics['profit_margin_pct'] > 20 else '#FFA500' if metrics['profit_margin_pct'] > 10 else '#EF553B'
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Profit Margin</h3>
                    <h1 style="color: {margin_color};">{metrics['profit_margin_pct']}%</h1>
                    <p>TTM</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # Detailed metrics tables
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💰 Valuation Metrics")
                valuation_df = pd.DataFrame({
                    'Metric': ['Current Price', 'P/E Ratio', 'Forward P/E', 'Market Cap', 'Beta'],
                    'Value': [
                        f"${metrics['price']}", 
                        f"{metrics['pe_ratio']}", 
                        f"{metrics['forward_pe']}",
                        f"${metrics['market_cap_b']}B", 
                        f"{metrics['beta']}"
                    ]
                })
                st.dataframe(valuation_df, hide_index=True, use_container_width=True)
            
            with col2:
                st.subheader("📈 Performance Metrics")
                performance_df = pd.DataFrame({
                    'Metric': ['Total Revenue', 'Revenue Growth', 'Profit Margin', 'Operating Margin', 'Debt/Equity'],
                    'Value': [
                        f"${metrics['revenue_b']}B",
                        f"{metrics['revenue_growth_pct']}%", 
                        f"{metrics['profit_margin_pct']}%",
                        f"{metrics['operating_margin_pct']}%",
                        f"{metrics['debt_to_equity']}"
                    ]
                })
                st.dataframe(performance_df, hide_index=True, use_container_width=True)
            
            st.divider()
            
            # Price chart
            st.subheader("📈 6-Month Price Chart")
            
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="6mo")
                
                if not hist.empty:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=hist.index, 
                        y=hist['Close'],
                        mode='lines',
                        name='Price',
                        line=dict(color='#FF6B35', width=2),
                        hovertemplate='$%{y:.2f}<extra></extra>'
                    ))
                    
                    # Add 20-day moving average if enough data
                    if len(hist) >= 20:
                        hist['MA20'] = hist['Close'].rolling(window=20).mean()
                        fig.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist['MA20'],
                            mode='lines',
                            name='20-Day MA',
                            line=dict(color='#00CC96', width=1, dash='dot')
                        ))
                    
                    fig.update_layout(
                        yaxis_title='Price ($)',
                        xaxis_title='Date',
                        template='plotly_dark',
                        height=450,
                        hovermode='x unified',
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Chart data unavailable (Yahoo Finance rate limited)")
                    
            except Exception as chart_error:
                st.warning(f"⚠️ Unable to load chart: {str(chart_error)}")
            
        else:
            st.error(f"⚠️ Unable to fetch data for {ticker}")
            st.info("💡 Available demo tickers: AAPL, META, MSFT, TSLA, GOOGL, NVDA, AMZN")
            
    except Exception as e:
        st.error(f"Error loading metrics: {str(e)}")

# ============================================================================
# TAB 3: ABOUT
# ============================================================================

with tab3:
    st.header("About FinChat AI")
    
    st.markdown("""
    ### 🎯 Multi-Agent Financial Analysis System
    
    **FinChat AI** uses specialized AI agents powered by Llama 3.1 70B to provide 
    comprehensive stock analysis from multiple perspectives.
    
    ### 🤖 Specialized Agents
    
    - **💼 Investment Advisor Agent** - Buy/sell/hold recommendations with specific entry/exit prices
    - **🛡️ Risk Analysis Agent** - Comprehensive risk assessment from metrics and SEC filings
    - **📦 Product Analysis Agent** - Business segment and product performance evaluation
    - **🏆 Peer Comparison Agent** - Competitive positioning analysis vs sector peers
    - **📊 General Analysis Agent** - Flexible analysis for any financial question
    
    ### 🔥 Technology Stack
    
    - **LLM:** Llama 3.1 70B via Fireworks AI
    - **Framework:** LangChain Multi-Agent System
    - **Embeddings:** OpenAI text-embedding-3-small
    - **Vector Database:** ChromaDB
    - **Data:** Yahoo Finance + SEC EDGAR
    - **Interface:** Streamlit
    
    ### 📊 Key Features
    
    ✅ Multi-agent orchestration for complex queries  
    ✅ RAG-powered SEC filing analysis  
    ✅ Real-time market data integration  
    ✅ Personalized recommendations based on risk profile  
    ✅ Fast responses (~90% cheaper than GPT-4)  
    
    ### 👥 Development Team
    
    **IST.688.M001.FALL25 - Building Human-Centered AI Applications**
    
    - Bhushan Jain
    - Samiksha Singh
    - Anjali Kalra
    - Shraddha Aher
    
    ### ⚖️ Disclaimer
    
    This tool provides AI-generated analysis for **educational purposes only**.
    It does not constitute financial advice. Always consult qualified financial
    advisors before making investment decisions.
    
    **Data Sources:**
    - Real-time metrics: Yahoo Finance API
    - SEC filings: User-uploaded documents
    - Analysis: Llama 3.1 70B via Fireworks AI
    """)
    
    st.divider()
    
    # Session statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Messages", len(st.session_state.messages))
    with col2:
        st.metric("Filings Uploaded", len(st.session_state.uploaded_filings))
    with col3:
        st.metric("Active Agents", "5")
    with col4:
        st.metric("Version", "2.0")
    
    st.divider()
    
    # Export functionality
    if len(st.session_state.messages) > 0:
        export_data = {
            'ticker': st.session_state.current_ticker,
            'timestamp': datetime.now().isoformat(),
            'portfolio_config': st.session_state.portfolio_config,
            'conversation': st.session_state.messages,
            'filings_uploaded': len(st.session_state.uploaded_filings)
        }
        
        st.download_button(
            label="📥 Export Conversation (JSON)",
            data=json.dumps(export_data, indent=2),
            file_name=f"finchat_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True
        )

# ============================================================================
# SIMPLE FOOTER (NO GRAY BOX!)
# ============================================================================

st.divider()

# Simple text footer
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style='text-align: center;'>
        <p style='color: #888; font-size: 0.9rem; margin-bottom: 0.3rem;'>
            <strong>FinChat AI v2.0</strong>
        </p>
        <p style='color: #999; font-size: 0.85rem; margin-bottom: 0.3rem;'>
            Multi-Agent System with Llama 3.1 70B (Fireworks AI) & LangChain
        </p>
        <p style='color: #aaa; font-size: 0.8rem;'>
            IST.688.M001.FALL25 - Building Human-Centered AI Applications
        </p>
    </div>
    """, unsafe_allow_html=True)
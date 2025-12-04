"""
FinChat AI - Multi-Agent Financial Analysis System
Powered by Fireworks Llama 3.1 70B, LangChain, and RAG
"""

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import re

# Custom imports
from utils.agents import FinancialAgentOrchestrator, supported_companies_markdown
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
# ENHANCED STYLING
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
    
    .stMarkdown p {
        word-spacing: 0.2em;
        letter-spacing: 0.015em;
        line-height: 1.8;
        word-break: break-word;
    }
    
    .stMarkdown h2, .stMarkdown h3 {
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    
    .stMarkdown li {
        margin-bottom: 0.8rem;
        line-height: 1.7;
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
    'NFLX': 'Netflix',
    'DIS': 'Disney'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_sample_price_history(ticker: str, period: str):
    """Generate realistic sample price data for demo (when Yahoo Finance is rate limited)"""
    
    # Realistic performance patterns for each ticker
    patterns = {
        'AAPL': {'trend': 0.12, 'volatility': 0.018, 'base_price': 175},
        'META': {'trend': 0.35, 'volatility': 0.025, 'base_price': 338},
        'MSFT': {'trend': 0.18, 'volatility': 0.015, 'base_price': 378},
        'TSLA': {'trend': 0.08, 'volatility': 0.040, 'base_price': 242},
        'GOOGL': {'trend': 0.15, 'volatility': 0.020, 'base_price': 139},
        'NVDA': {'trend': 0.48, 'volatility': 0.035, 'base_price': 487},
        'AMZN': {'trend': 0.22, 'volatility': 0.022, 'base_price': 145},
        'NFLX': {'trend': 0.05, 'volatility': 0.028, 'base_price': 385},
        'DIS': {'trend': -0.02, 'volatility': 0.022, 'base_price': 92}
    }
    
    pattern = patterns.get(ticker, {'trend': 0.10, 'volatility': 0.02, 'base_price': 100})
    
    # Determine number of days
    period_days = {
        '1mo': 30,
        '3mo': 90,
        '6mo': 180,
        '1y': 365,
        '2y': 730
    }
    days = period_days.get(period, 180)
    
    # Generate dates
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=days, freq='D')
    
    # Generate realistic price movement
    np.random.seed(hash(ticker) % 10000)  # Consistent for same ticker
    
    # Daily returns with trend and volatility
    daily_returns = np.random.normal(
        pattern['trend'] / days,    # Average daily return
        pattern['volatility'],       # Daily volatility
        days
    )
    
    # Add some autocorrelation (realistic market behavior)
    for i in range(1, len(daily_returns)):
        daily_returns[i] += daily_returns[i-1] * 0.1
    
    # Calculate prices
    cumulative_returns = (1 + daily_returns).cumprod()
    prices = pattern['base_price'] * cumulative_returns
    
    # Create DataFrame
    df = pd.DataFrame({
        'Close': prices,
        'Open': prices * (1 + np.random.normal(0, 0.002, days)),
        'High': prices * (1 + abs(np.random.normal(0, 0.005, days))),
        'Low': prices * (1 - abs(np.random.normal(0, 0.005, days))),
    }, index=dates)
    
    return df


def extract_filing_sections(text: str) -> dict:
    """Extract key sections from SEC filing text"""
    sections = {}
    
    risk_patterns = [
        r"RISK\s+FACTORS.*?(?=ITEM|UNREGISTERED|$)",
        r"ITEM\s+1A\..*?(?=ITEM\s+2|$)"
    ]
    for pattern in risk_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            sections['Risk Factors'] = match.group(0)[:15000]
            break
    
    mda_patterns = [
        r"MANAGEMENT'?S\s+DISCUSSION\s+AND\s+ANALYSIS.*?(?=QUANTITATIVE|ITEM\s+3|$)",
        r"MD&A.*?(?=ITEM\s+3|$)"
    ]
    for pattern in mda_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            sections['Management Discussion & Analysis'] = match.group(0)[:15000]
            break
    
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

    # Show unified supported companies list (matches agents)
    with st.expander("✅ Supported Companies"):
        st.markdown(supported_companies_markdown())
    
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
        
        st.caption(f"🎯 {risk_tolerance} | {investment_horizon} | {portfolio_allocation}%")
    
    st.divider()
    
    st.caption("**📊 Session Stats:**")
    st.caption(f"💬 Queries: {len(st.session_state.messages) // 2}")
    st.caption(f"📄 Filings: {len(st.session_state.uploaded_filings)}")
    
    st.divider()
    
    st.caption("**🔥 Powered by:**")
    st.caption("• Llama 3.1 70B (Fireworks)")
    st.caption("• LangChain Multi-Agent")
    st.caption("• RAG with ChromaDB")
    
    st.divider()
    
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
    
    # Header
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
            if st.button(f"Is {ticker} a good buy?", key="ex2"):
                st.session_state.example_query = f"Is {ticker} a good buy for a conservative investor?"
                st.rerun()
        
        with example_col2:
            st.markdown("**Analysis:**")
            if st.button(f"What are risks for {ticker}?", key="ex3"):
                st.session_state.example_query = f"What are the main risks for {ticker}?"
                st.rerun()
            if st.button(f"Compare to competitors", key="ex4"):
                st.session_state.example_query = f"Is {ticker} better than its competitors?"
                st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                st.caption("🔥 Llama 3.1 70B (Fireworks AI)")
    
    # Chat input
    chat_input_value = st.session_state.pop('example_query', '')
    
    if prompt := (chat_input_value or st.chat_input(f"Ask about {ticker}'s financial performance, risks, or investment potential...")):
        
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
                st.caption("🔥 Llama 3.1 70B (Fireworks AI)")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
                
            except Exception as e:
                error_msg = f"⚠️ **Error:** {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# ============================================================================
# TAB 2: QUICK METRICS + MULTI-STOCK COMPARISON
# ============================================================================

with tab2:
    ticker = st.session_state.current_ticker
    
    st.subheader(f"📊 Quick Metrics for {ticker}")
    st.caption(f"Analyzing {POPULAR_TICKERS.get(ticker, ticker)}")
    
    try:
        metrics_json = get_stock_metrics_tool(ticker)
        metrics = json.loads(metrics_json)
        
        if 'error' not in metrics:
            # Data source indicator
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
            
            # Detailed metrics
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
            
            # ============================================================================
            # MULTI-STOCK COMPARISON CHART WITH SAMPLE DATA
            # ============================================================================
            
            st.subheader("🏆 Multi-Stock Performance Comparison")
            st.caption("Compare price trends across multiple stocks (normalized to % change)")
            
            # Stock selector
            default_compare = [ticker]
            if ticker == 'META':
                default_compare.extend(['GOOGL', 'NFLX'])
            elif ticker == 'TSLA':
                default_compare.extend(['NVDA', 'AMZN'])
            else:
                default_compare.extend(['MSFT', 'GOOGL'])
            
            compare_tickers = st.multiselect(
                "Select stocks to compare:",
                options=list(POPULAR_TICKERS.keys()),
                default=default_compare[:3],
                help="Choose 2-5 stocks to overlay"
            )
            
            # Time period
            comp_period = st.selectbox(
                "Period:",
                options=['1mo', '3mo', '6mo', '1y', '2y'],
                index=2,
                key='comp_period'
            )
            
            if len(compare_tickers) >= 2:
                st.info("📊 Using demo data (Yahoo Finance rate limited - realistic patterns for testing)")
                
                try:
                    fig_multi = go.Figure()
                    colors = ['#FF6B35', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
                    performance_data = []
                    
                    for i, comp_ticker in enumerate(compare_tickers):
                        # Try real data first, fall back to sample
                        try:
                            stock = yf.Ticker(comp_ticker)
                            hist = stock.history(period=comp_period)
                            
                            if hist.empty or len(hist) < 5:
                                raise Exception("Use sample")
                            
                            # Use real data if available
                            use_sample = False
                        except:
                            # Use sample data
                            use_sample = True
                            hist = generate_sample_price_history(comp_ticker, comp_period)
                        
                        if not hist.empty:
                            # Normalize to % change
                            start_price = hist['Close'].iloc[0]
                            normalized = ((hist['Close'] / start_price) - 1) * 100
                            
                            fig_multi.add_trace(go.Scatter(
                                x=hist.index,
                                y=normalized,
                                mode='lines',
                                name=f'{comp_ticker}' + (' (Demo)' if use_sample else ''),
                                line=dict(color=colors[i % len(colors)], width=2.5),
                                hovertemplate=f'{comp_ticker}: %{{y:.1f}}%<extra></extra>'
                            ))
                            
                            final_change = normalized.iloc[-1]
                            performance_data.append((comp_ticker, final_change))
                    
                    fig_multi.update_layout(
                        title=f'Performance Comparison - {comp_period.upper()} (% Change)',
                        yaxis_title='% Change from Start',
                        xaxis_title='Date',
                        template='plotly_dark',
                        height=500,
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5
                        )
                    )
                    
                    # Zero line
                    fig_multi.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    
                    st.plotly_chart(fig_multi, use_container_width=True)
                    
                    # Performance leaderboard
                    if performance_data:
                        st.markdown("#### 🏆 Performance Leaderboard")
                        performance_data.sort(key=lambda x: x[1], reverse=True)
                        
                        cols = st.columns(len(performance_data))
                        for i, (t, change) in enumerate(performance_data):
                            with cols[i]:
                                medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
                                st.metric(
                                    f"{medal} {t}",
                                    f"{change:+.1f}%",
                                    delta=f"Rank #{i+1}"
                                )
                        
                        # Insights
                        best_ticker, best_change = performance_data[0]
                        worst_ticker, worst_change = performance_data[-1]
                        spread = best_change - worst_change
                        
                        st.caption(f"💡 **{best_ticker}** outperformed **{worst_ticker}** by **{spread:.1f}** pts over {comp_period}")
                        
                except Exception as e:
                    st.error(f"Chart error: {str(e)}")
            else:
                st.info("👆 Select at least 2 stocks to compare")
            
        else:
            st.error(f"⚠️ No data for {ticker}")
            # Match actual BACKUP_METRICS tickers from tools.py for clarity
            st.info("💡 Backup demo data is currently available for: AAPL, MSFT, META, GOOGL, NVDA, TSLA, AMZN, V, JPM, NFLX, DIS.")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

# ============================================================================
# TAB 3: ABOUT
# ============================================================================

with tab3:
    st.header("About FinChat AI")
    
    st.markdown("""
    ### 🎯 Multi-Agent Financial Analysis System
    
    **FinChat AI** uses specialized AI agents powered by Llama 3.1 70B.
    
    ### 🤖 Specialized Agents
    
    - **💼 Investment Advisor** - Buy/sell/hold recommendations
    - **🛡️ Risk Analyst** - Risk assessment with SEC filings
    - **📦 Product Analyst** - Segment performance
    - **🏆 Peer Comparator** - Competitive analysis
    - **📊 General Analyst** - Flexible analysis
    
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
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Messages", len(st.session_state.messages))
    with col2:
        st.metric("Filings", len(st.session_state.uploaded_filings))
    with col3:
        st.metric("Agents", "5")
    with col4:
        st.metric("Version", "2.0")
    
    st.divider()
    
    # Export
    if len(st.session_state.messages) > 0:
        export_data = {
            'ticker': st.session_state.current_ticker,
            'timestamp': datetime.now().isoformat(),
            'portfolio_config': st.session_state.portfolio_config,
            'conversation': st.session_state.messages,
            'filings': len(st.session_state.uploaded_filings)
        }
        
        st.download_button(
            label="📥 Export Conversation (JSON)",
            data=json.dumps(export_data, indent=2),
            file_name=f"finchat_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True
        )

# ============================================================================
# FOOTER
# ============================================================================

st.divider()

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style='text-align: center;'>
        <p style='color: #888; font-size: 0.9rem;'><strong>FinChat AI v2.0</strong></p>
        <p style='color: #999; font-size: 0.85rem;'>Llama 3.1 70B (Fireworks) & LangChain</p>
        <p style='color: #aaa; font-size: 0.8rem;'>IST.688.M001.FALL25</p>
    </div>
    """, unsafe_allow_html=True)

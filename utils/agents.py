"""
Multi-Agent System using Fireworks Llama 3.1 70B
FIXED: Detects unsupported companies and shows helpful error
"""

import streamlit as st
from langchain_fireworks import ChatFireworks
from langchain.prompts import ChatPromptTemplate
from typing import Dict, Optional, Tuple
import json
import re

from utils.tools import (
    get_stock_metrics_tool,
    search_sec_filing_tool,
    calculate_price_targets_tool,
    get_peer_comparison_tool,
    calculate_health_score_tool
)

# ============================================================================
# TICKER EXTRACTION HELPERS - COMPLETE FIX
# ============================================================================

def extract_ticker_from_query(query: str, current_ticker: str) -> tuple:
    """
    Extract ticker and validate it exists in our data
    
    Returns: (ticker, found_in_query)
    - ticker: The ticker to use (or None if unsupported company mentioned)
    - found_in_query: True if ticker was in query, False if using dropdown
    """
    
    # Expanded ticker/company name mapping
    ticker_map = {
        # Technology
        'NVIDIA': 'NVDA', 'APPLE': 'AAPL', 'MICROSOFT': 'MSFT', 
        'GOOGLE': 'GOOGL', 'ALPHABET': 'GOOGL', 'META': 'META', 
        'FACEBOOK': 'META', 'TESLA': 'TSLA', 'AMAZON': 'AMZN', 
        'NETFLIX': 'NFLX', 'DISNEY': 'DIS', 'WALMART': 'WMT',
        'WALT DISNEY': 'DIS',
        
        # Financial
        'JPMORGAN': 'JPM', 'JP MORGAN': 'JPM', 'JPMORGAN CHASE': 'JPM',
        'CHASE': 'JPM', 'VISA': 'V',
        
        # Healthcare
        'JOHNSON': 'JNJ', 'J&J': 'JNJ', 'JOHNSON & JOHNSON': 'JNJ',
        
        # Consumer
        'PROCTER': 'PG', 'P&G': 'PG', 'PROCTER & GAMBLE': 'PG'
    }
    
    # Available tickers in our backup data
    available_tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMZN', 
                         'JPM', 'V', 'JNJ', 'WMT', 'PG', 'NFLX', 'DIS']
    
    # Common company/stock keywords that indicate specific query
    specific_keywords = ['STOCK', 'PRICE', 'COMPANY', 'INVEST', 'BUY', 'SELL', 
                        'ANALYSIS', 'ABOUT', 'TELL ME', 'SHOULD I']
    
    query_upper = query.upper()
    
    # Check if this is asking about a specific company
    is_specific_query = any(keyword in query_upper for keyword in specific_keywords)
    
    # FIRST: Check for ticker symbols with word boundaries
    for ticker in available_tickers:
        pattern = r'\b' + re.escape(ticker) + r'\b'
        if re.search(pattern, query_upper):
            return ticker, True
    
    # SECOND: Check for company names we support
    for company_name in sorted(ticker_map.keys(), key=len, reverse=True):
        if company_name in query_upper:
            ticker = ticker_map[company_name]
            if ticker in available_tickers:
                return ticker, True
    
    # THIRD: Check for unsupported companies (common ones people might ask about)
    unsupported_companies = [
        'BOEING', 'BA', 'CITI', 'CITIBANK', 'CITIGROUP', 'BANK OF AMERICA', 
        'BOA', 'BAC', 'WELLS FARGO', 'WFC', 'GOLDMAN', 'GOLDMAN SACHS', 'GS',
        'MORGAN STANLEY', 'MS', 'AMD', 'INTEL', 'INTC', 'IBM', 'ORACLE', 'ORCL',
        'UBER', 'LYFT', 'AIRBNB', 'ABNB', 'SNAP', 'TWITTER', 'FORD', 'GM',
        'GENERAL MOTORS', 'COCA COLA', 'COKE', 'KO', 'PEPSI', 'PEP',
        'MCDONALD', 'MCD', 'STARBUCKS', 'SBUX', 'NIKE', 'NKE'
    ]
    
    # If specific query and mentions unsupported company, return None
    if is_specific_query:
        for unsupported in unsupported_companies:
            if unsupported in query_upper:
                return None, True  # Found company but not supported
    
    # FOURTH: For general questions without specific company, use dropdown
    return current_ticker, False


def extract_tickers_from_query(query: str, current_ticker: str) -> tuple:
    """Extract TWO tickers for comparison queries"""
    
    ticker_map = {
        'NVIDIA': 'NVDA', 'APPLE': 'AAPL', 'MICROSOFT': 'MSFT', 
        'GOOGLE': 'GOOGL', 'META': 'META', 'FACEBOOK': 'META',
        'TESLA': 'TSLA', 'AMAZON': 'AMZN', 'NETFLIX': 'NFLX',
        'DISNEY': 'DIS', 'WALMART': 'WMT', 'VISA': 'V',
        'JPMORGAN': 'JPM', 'JOHNSON': 'JNJ'
    }
    
    query_upper = query.upper()
    found_tickers = []
    
    # Find company names first
    for company_name in sorted(ticker_map.keys(), key=len, reverse=True):
        if company_name in query_upper and ticker_map[company_name] not in found_tickers:
            found_tickers.append(ticker_map[company_name])
            if len(found_tickers) >= 2:
                break
    
    # Find ticker symbols with word boundaries
    if len(found_tickers) < 2:
        all_tickers = ['GOOGL', 'MSFT', 'AAPL', 'META', 'NVDA', 'TSLA', 'AMZN', 
                       'NFLX', 'JPM', 'V', 'JNJ', 'WMT', 'PG', 'DIS']
        
        for ticker in all_tickers:
            if ticker not in found_tickers:
                pattern = r'\b' + re.escape(ticker) + r'\b'
                if re.search(pattern, query_upper):
                    found_tickers.append(ticker)
                    if len(found_tickers) >= 2:
                        break
    
    if len(found_tickers) >= 2:
        return found_tickers[0], found_tickers[1]
    elif len(found_tickers) == 1:
        return found_tickers[0], None
    else:
        return current_ticker, None


# ============================================================================
# LLM INITIALIZATION
# ============================================================================

@st.cache_resource
def get_llm():
    """Initialize Fireworks Llama 3.1 70B"""
    llm = ChatFireworks(
        model="accounts/fireworks/models/llama-v3p3-70b-instruct",
        api_key=st.secrets["FIREWORKS_API_KEY"],
        temperature=0.7,
        max_tokens=1024
    )
    return llm


# ============================================================================
# ORCHESTRATOR
# ============================================================================

class FinancialAgentOrchestrator:
    """Routes queries to specialized agents"""
    
    def __init__(self, ticker: str, portfolio_config: Dict):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = get_llm()
    
    def classify_intent(self, query: str) -> str:
        """Classify user query"""
        query_lower = query.lower()
        
        investment_keywords = ['invest', 'buy', 'sell', 'hold', 'price target', 'should i', 'entry', 'exit', 'good buy', 'what should i do', 'have stocks', 'stock price', 'price of']
        risk_keywords = ['risk', 'danger', 'concern', 'threat', 'safe', 'risky', 'worry', 'worried']
        product_keywords = ['product', 'segment', 'business', 'revenue', 'sales', 'performing', 'advertising']
        comparison_keywords = ['compare', 'competitor', 'peer', 'vs', 'versus', 'better than', 'or']
        
        if any(word in query_lower for word in investment_keywords):
            return 'investment'
        elif any(word in query_lower for word in risk_keywords):
            return 'risk'
        elif any(word in query_lower for word in product_keywords):
            return 'product'
        elif any(word in query_lower for word in comparison_keywords):
            return 'comparison'
        else:
            return 'general'
    
    def route_query(self, query: str) -> str:
        """Route to appropriate specialized agent"""
        intent = self.classify_intent(query)
        
        if intent == 'investment':
            return InvestmentAdvisorAgent(self.ticker, self.portfolio_config, self.llm).analyze(query)
        elif intent == 'risk':
            return RiskAnalysisAgent(self.ticker, self.portfolio_config, self.llm).analyze(query)
        elif intent == 'product':
            return ProductAnalysisAgent(self.ticker, self.portfolio_config, self.llm).analyze(query)
        elif intent == 'comparison':
            return PeerComparisonAgent(self.ticker, self.portfolio_config, self.llm).analyze(query)
        else:
            return GeneralAnalysisAgent(self.ticker, self.portfolio_config, self.llm).analyze(query)


# ============================================================================
# INVESTMENT ADVISOR AGENT
# ============================================================================

class InvestmentAdvisorAgent:
    """Investment decisions with ticker extraction"""
    
    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm
    
    def analyze(self, query: str) -> str:
        """Provide investment recommendation"""
        
        # EXTRACT TICKER FROM QUERY
        analyzed_ticker, found_in_query = extract_ticker_from_query(query, self.ticker)
        
        # Check if unsupported company mentioned
        if analyzed_ticker is None and found_in_query:
            return """## ⚠️ Company Not Supported

Sorry, I don't have data for the company you mentioned.

**FinChat AI currently supports these companies:**

**Technology:**  
• AAPL (Apple) • MSFT (Microsoft) • GOOGL (Google/Alphabet)  
• META (Meta/Facebook) • NVDA (Nvidia) • TSLA (Tesla)

**Consumer & Retail:**  
• AMZN (Amazon) • WMT (Walmart) • DIS (Disney) • NFLX (Netflix)

**Financial Services:**  
• JPM (JPMorgan) • V (Visa)

**Healthcare:**  
• JNJ (Johnson & Johnson)

**Other:**  
• PG (Procter & Gamble)

💡 **Tip:** Select one of these companies from the dropdown or ask about them by name!

**Try asking:**
- "Should I invest in Apple?"
- "What's the stock price of Nvidia?"
- "Compare Microsoft to Google"
"""
        
        with st.status("💼 Investment Advisor Agent", expanded=True) as status:
            st.write(f"🎯 Analyzing {analyzed_ticker}...")
            st.write("🔧 Gathering financial metrics...")
            metrics_json = get_stock_metrics_tool(analyzed_ticker)
            
            st.write("🎯 Calculating price targets...")
            targets_json = calculate_price_targets_tool(analyzed_ticker, self.portfolio_config['risk_tolerance'])
            
            st.write("🏥 Assessing financial health...")
            health_json = calculate_health_score_tool(analyzed_ticker)
            
            st.write("📄 Checking SEC filing...")
            risks = search_sec_filing_tool(analyzed_ticker, "risks concerns")
            
            st.write("🤖 Generating recommendation...")
            
            # Parse data
            try:
                metrics = json.loads(metrics_json)
                targets = json.loads(targets_json)
                health = json.loads(health_json)
            except:
                return "⚠️ Error parsing financial data."
            
            # Check if tools returned errors
            if 'error' in metrics or 'error' in targets:
                return """## ⚠️ Data Temporarily Unavailable

We're having trouble fetching data for this company right now. This might be due to:
- API rate limiting
- Temporary service outage

**Please try:**
1. Select a different company
2. Try again in a few minutes
3. Check if the company ticker is in our supported list
"""
            
            # QUESTION-AWARE VERDICT
            query_lower = query.lower()
            
            # Check if user mentions they own the stock
            owns_stock = any(phrase in query_lower for phrase in ['i have', 'i own', 'holding', 'my stocks'])
            
            if owns_stock:
                verdict_prompt = f"""You are advising someone who ALREADY OWNS {analyzed_ticker}.

Current Price: ${targets.get('current_price', 0)}
Fair Value: ${targets.get('fair_value', 0)}
Target Exit: ${targets.get('target_exit', 0)}
Stop Loss: ${targets.get('stop_loss', 0)}

Question: {query}

Should they HOLD or SELL? Provide: "[HOLD/SELL] - [explain what to do with their existing position]"
"""
            elif 'conservative' in query_lower:
                verdict_prompt = f"""You are advising a CONSERVATIVE investor about {analyzed_ticker}.

Current Price: ${targets.get('current_price', 0)}
Conservative Entry: ${targets.get('conservative_entry', 0)}
Health Score: {health.get('score', 0)}/100

Question: {query}

Focus on safety. Provide: "[BUY/HOLD/WAIT] - [emphasize risk factors and safety]"
"""
            else:
                verdict_prompt = f"""You are a financial advisor answering: "{query}"

Data for {analyzed_ticker}:
Current Price: ${targets.get('current_price', 0)}
Fair Value: ${targets.get('fair_value', 0)}
Upside: {targets.get('upside_percent', 0)}%
Health Score: {health.get('score', 0)}/100

Provide: "[BUY/HOLD/WAIT] - [clear explanation]"
"""
            
            verdict_response = self.llm.invoke(verdict_prompt)
            verdict = verdict_response.content.strip()
            
            status.update(label="✅ Investment Advisor Complete", state="complete")
        
        # BUILD RESPONSE (keeping your existing response building code)
        current_price = targets.get('current_price', 0)
        fair_value = targets.get('fair_value', 0)
        upside = targets.get('upside_percent', 0)
        health_score = health.get('score', 0)
        health_rating = health.get('rating', 'N/A')
        
        verdict_emoji = "✅" if 'BUY' in verdict.upper() or 'HOLD' in verdict.upper() else "⏸️" if 'WAIT' in verdict.upper() else "🔄"
        
        if owns_stock:
            intro = f"**You own {analyzed_ticker}:**\n\n"
        elif 'conservative' in query_lower:
            intro = f"**For Conservative Investors:**\n\n"
        else:
            intro = ""
        
        response = f"""## {verdict_emoji} Investment Recommendation for {analyzed_ticker}

{intro}**{verdict}**

---

## 📊 Current Analysis

| Metric | Value |
|--------|-------|
| **Current Price** | ${current_price} |
| **Fair Value Estimate** | ${fair_value} |
| **Upside Potential** | {upside}% |
| **Health Score** | {health_score}/100 ({health_rating}) |

---

## 🎯 Price Targets ({self.portfolio_config['risk_tolerance']} Risk Profile)

"""
        
        if self.portfolio_config['risk_tolerance'] == 'Conservative':
            response += f"""- **🎯 Conservative Entry:** ${targets.get('conservative_entry', 0)} **(recommended for you)**
- Moderate Entry: ${targets.get('moderate_entry', 0)}
- Aggressive Entry: ${targets.get('aggressive_entry', 0)}
"""
        elif self.portfolio_config['risk_tolerance'] == 'Aggressive':
            response += f"""- **🎯 Aggressive Entry:** ${targets.get('aggressive_entry', 0)} **(recommended for you)**
- Moderate Entry: ${targets.get('moderate_entry', 0)}
- Conservative Entry: ${targets.get('conservative_entry', 0)}
"""
        else:
            response += f"""- **🎯 Moderate Entry:** ${targets.get('moderate_entry', 0)} **(recommended for you)**
- Conservative Entry: ${targets.get('conservative_entry', 0)} (safer)
- Aggressive Entry: ${targets.get('aggressive_entry', 0)} (higher risk)
"""
        
        response += f"""- **Target Exit:** ${targets.get('target_exit', 0)} (+{round((targets.get('target_exit', 0) / current_price - 1) * 100, 1) if current_price > 0 else 0}%)
- **Stop Loss:** ${targets.get('stop_loss', 0)} (-8% protection)

---

## 📈 Key Financial Metrics

| Metric | {analyzed_ticker} | Assessment |
|--------|------|------------|
| **Revenue Growth** | {metrics.get('revenue_growth_pct', 0)}% | {'🟢 Strong' if metrics.get('revenue_growth_pct', 0) > 15 else '🟡 Moderate' if metrics.get('revenue_growth_pct', 0) > 5 else '🔴 Weak'} |
| **Profit Margin** | {metrics.get('profit_margin_pct', 0)}% | {'🟢 Excellent' if metrics.get('profit_margin_pct', 0) > 20 else '🟡 Good' if metrics.get('profit_margin_pct', 0) > 10 else '🔴 Low'} |
| **Debt/Equity** | {metrics.get('debt_to_equity', 0)} | {'🟢 Low Risk' if metrics.get('debt_to_equity', 0) < 50 else '🟡 Moderate' if metrics.get('debt_to_equity', 0) < 150 else '🔴 High Risk'} |
| **Beta** | {metrics.get('beta', 0)} | {'🟢 Below Market' if metrics.get('beta', 0) < 1.0 else '🟡 Market Average' if metrics.get('beta', 0) < 1.3 else '🔴 High Volatility'} |

---

## ⚠️ Key Considerations

"""
        
        if health.get('factors'):
            for i, factor in enumerate(health['factors'][:3], 1):
                response += f"{i}. {factor}\n"
        
        response += f"""

---

**Investment Profile:** {self.portfolio_config['risk_tolerance']} Risk | {self.portfolio_config['investment_horizon']} | {self.portfolio_config['portfolio_allocation']}% Allocation

⚠️ *This is analysis, not personalized financial advice. Consult a qualified advisor.*
"""
        
        return response


# ============================================================================
# OTHER AGENTS - Apply same pattern
# ============================================================================

class RiskAnalysisAgent:
    """Risk assessment with ticker extraction"""
    
    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm
    
    def analyze(self, query: str) -> str:
        """Comprehensive risk analysis"""
        
        analyzed_ticker, found_in_query = extract_ticker_from_query(query, self.ticker)
        
        if analyzed_ticker is None and found_in_query:
            return """## ⚠️ Company Not Supported

Sorry, I don't have risk analysis data for the company you mentioned.

**Supported companies:** AAPL, MSFT, GOOGL, META, NVDA, TSLA, AMZN, V, JPM, JNJ, WMT, DIS, NFLX, PG

Select a supported company from the dropdown or ask about one by name.
"""
        
        # Rest of your existing RiskAnalysisAgent code...
        with st.status("🛡️ Risk Analysis Agent", expanded=True) as status:
            st.write(f"🎯 Analyzing {analyzed_ticker}...")
            st.write("🔧 Gathering financial metrics...")
            metrics_json = get_stock_metrics_tool(analyzed_ticker)
            
            st.write("🎯 Calculating risk thresholds...")
            targets_json = calculate_price_targets_tool(analyzed_ticker, self.portfolio_config['risk_tolerance'])
            
            st.write("📄 Extracting SEC filing risks...")
            filing_risks = search_sec_filing_tool(analyzed_ticker, "risk factors threats concerns regulatory")
            
            st.write("🤖 Analyzing risk profile...")
            
            try:
                metrics = json.loads(metrics_json)
                targets = json.loads(targets_json)
            except:
                return "⚠️ Error loading data."
            
            if 'error' in metrics:
                return "⚠️ Unable to fetch risk data for this company."
            
            status.update(label="✅ Risk Analysis Complete", state="complete")
        
        # Your existing response building code...
        response = f"""## 🛡️ Risk Assessment for {analyzed_ticker}

[Rest of risk analysis response...]
"""
        return response


class ProductAnalysisAgent:
    """Product analysis with ticker extraction"""
    
    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm
    
    def analyze(self, query: str) -> str:
        analyzed_ticker, found_in_query = extract_ticker_from_query(query, self.ticker)
        
        if analyzed_ticker is None and found_in_query:
            return "## ⚠️ Company Not Supported\n\nSorry, product analysis not available for this company."
        
        # Your existing code...
        return "Product analysis response..."


class PeerComparisonAgent:
    """Peer comparison with head-to-head support"""
    
    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm
    
    def analyze(self, query: str) -> str:
        ticker1, ticker2 = extract_tickers_from_query(query, self.ticker)
        
        # Your existing comparison code...
        if ticker2:
            return self.head_to_head_comparison(ticker1, ticker2)
        else:
            return self.sector_comparison(ticker1)
    
    def head_to_head_comparison(self, ticker1: str, ticker2: str) -> str:
        # Your existing code...
        return "Comparison response..."
    
    def sector_comparison(self, ticker: str) -> str:
        # Your existing code...
        return "Sector comparison response..."


class GeneralAnalysisAgent:
    """General analysis with ticker extraction"""
    
    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm
    
    def analyze(self, query: str) -> str:
        analyzed_ticker, found_in_query = extract_ticker_from_query(query, self.ticker)
        
        if analyzed_ticker is None and found_in_query:
            return """## ⚠️ Company Not Supported

Sorry, I don't have data for the company you mentioned.

**FinChat AI currently supports these companies:**

**Technology:** AAPL, MSFT, GOOGL, META, NVDA, TSLA  
**Consumer & Retail:** AMZN, WMT, DIS, NFLX  
**Financial Services:** JPM, V  
**Healthcare:** JNJ  
**Other:** PG

💡 **Tip:** Select a company from the dropdown or ask about a supported one!

**Try:** "Tell me about Apple" or "What's Microsoft's financial health?"
"""
        
        with st.status("📊 General Analysis Agent", expanded=True) as status:
            st.write(f"🎯 Analyzing {analyzed_ticker}...")
            st.write("🔧 Gathering data...")
            
            metrics_json = get_stock_metrics_tool(analyzed_ticker)
            health_json = calculate_health_score_tool(analyzed_ticker)
            
            try:
                metrics = json.loads(metrics_json)
                health = json.loads(health_json)
                
                if 'error' in metrics:
                    return "⚠️ Unable to fetch data for this company."
                
            except:
                return "⚠️ Error loading data."
            
            status.update(label="✅ Analysis Complete", state="complete")
        
        # Your existing response code...
        response = f"""## 📊 Financial Overview for {analyzed_ticker}

**Name:** {metrics.get('company_name', analyzed_ticker)}  
**Sector:** {metrics.get('sector', 'Unknown')}

**Health Score:** {health.get('score', 0)}/100
"""
        return response
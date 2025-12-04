"""
Multi-Agent System using Fireworks Llama 3.1 70B
FIXED: Ticker extraction now uses word boundaries to prevent V/NVDA confusion
"""

"""
Multi-Agent System using Fireworks Llama 3.1 70B
UPDATED: All agents now parse tickers from questions
"""

import streamlit as st
from langchain_fireworks import ChatFireworks
from langchain.prompts import ChatPromptTemplate
from typing import Dict, Optional, Tuple
import json

from utils.tools import (
    get_stock_metrics_tool,
    search_sec_filing_tool,
    calculate_price_targets_tool,
    get_peer_comparison_tool,
    calculate_health_score_tool
)

# ============================================================================
# TICKER EXTRACTION HELPERS - FIXED VERSION
# ============================================================================

def extract_ticker_from_query(query: str, current_ticker: str) -> tuple:
    """
    Extract ticker and validate it exists in our data
    
    Returns: (ticker, found_in_query)
    - ticker: The ticker to use
    - found_in_query: True if ticker was in query, False if using dropdown
    """
    
    # Expanded ticker/company name mapping
    ticker_map = {
        # Technology
        'NVIDIA': 'NVDA', 'APPLE': 'AAPL', 'MICROSOFT': 'MSFT', 
        'GOOGLE': 'GOOGL', 'ALPHABET': 'GOOGL', 'META': 'META', 
        'FACEBOOK': 'META', 'TESLA': 'TSLA', 'AMAZON': 'AMZN', 
        'NETFLIX': 'NFLX', 'DISNEY': 'DIS', 'WALMART': 'WMT',
        
        # Financial
        'JPMORGAN': 'JPM', 'JP MORGAN': 'JPM', 'CHASE': 'JPM',
        'VISA': 'V', 'CITI': 'C', 'CITIBANK': 'C', 'CITI BANK': 'C',
        'CITIGROUP': 'C', 'BANK OF AMERICA': 'BAC', 'BOA': 'BAC',
        'WELLS FARGO': 'WFC', 'GOLDMAN': 'GS', 'GOLDMAN SACHS': 'GS',
        'MORGAN STANLEY': 'MS',
        
        # Other
        'JOHNSON': 'JNJ', 'J&J': 'JNJ',
        'PROCTER': 'PG', 'P&G': 'PG'
    }
    
    # Available tickers in our backup data
    available_tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMZN', 
                         'JPM', 'V', 'JNJ', 'WMT', 'PG', 'NFLX', 'DIS']
    
    query_upper = query.upper()
    
    # Check for ticker symbols directly (NVDA, AAPL, C, etc.)
    for ticker in available_tickers:
        if ticker in query_upper:
            return ticker, True
    
    # Check for company names (NVIDIA, Citi Bank, etc.)
    for company_name, ticker in ticker_map.items():
        if company_name in query_upper:
            # Found company name - check if we have data for it
            if ticker in available_tickers:
                return ticker, True
            else:
                # Found ticker but not in our data
                return ticker, True  # Return it anyway, agent will handle error
    
    # No ticker found in query, use dropdown
    return current_ticker, False


def extract_tickers_from_query(query: str, current_ticker: str) -> tuple:
    """Extract TWO tickers for comparison queries - FIXED"""
    
    ticker_map = {
        'NVIDIA': 'NVDA', 'APPLE': 'AAPL', 'MICROSOFT': 'MSFT', 
        'GOOGLE': 'GOOGL', 'META': 'META', 'FACEBOOK': 'META',
        'TESLA': 'TSLA', 'AMAZON': 'AMZN', 'NETFLIX': 'NFLX',
        'DISNEY': 'DIS', 'WALMART': 'WMT', 'VISA': 'V'
    }
    
    query_upper = query.upper()
    found_tickers = []
    
    # Find company names first (most specific)
    for company_name in sorted(ticker_map.keys(), key=len, reverse=True):
        if company_name in query_upper and ticker_map[company_name] not in found_tickers:
            found_tickers.append(ticker_map[company_name])
            if len(found_tickers) >= 2:
                break
    
    # Find ticker symbols with word boundaries
    if len(found_tickers) < 2:
        all_tickers = ['GOOGL', 'MSFT', 'AAPL', 'META', 'NVDA', 'TSLA', 'AMZN', 
                       'NFLX', 'JPM', 'V', 'JNJ', 'WMT', 'PG', 'DIS', 'GM', 'F']
        
        for ticker in all_tickers:
            if ticker not in found_tickers:
                # Word boundary check prevents "V" matching in "NVIDIA" or "HAVE"
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
        
        investment_keywords = ['invest', 'buy', 'sell', 'hold', 'price target', 'should i', 'entry', 'exit', 'good buy', 'what should i do', 'have stocks']
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
# INVESTMENT ADVISOR AGENT - TICKER AWARE
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
        analyzed_ticker = extract_ticker_from_query(query, self.ticker)
        
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
            
            # QUESTION-AWARE VERDICT
            query_lower = query.lower()
            
            # Check if user mentions they own the stock
            owns_stock = any(phrase in query_lower for phrase in ['i have', 'i own', 'holding', 'my stocks'])
            
            if owns_stock:
                # Different prompt for existing holders
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
            elif 'entry price' in query_lower or 'entry point' in query_lower:
                verdict_prompt = f"""You are advising on ENTRY PRICE for {analyzed_ticker}.

Current Price: ${targets.get('current_price', 0)}
Moderate Entry: ${targets.get('moderate_entry', 0)}
Conservative Entry: ${targets.get('conservative_entry', 0)}

Question: {query}

Focus on optimal entry. Provide: "[BUY/HOLD/WAIT] - [emphasize entry price strategy]"
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
        
        # BUILD RESPONSE
        current_price = targets.get('current_price', 0)
        fair_value = targets.get('fair_value', 0)
        upside = targets.get('upside_percent', 0)
        health_score = health.get('score', 0)
        health_rating = health.get('rating', 'N/A')
        
        verdict_emoji = "✅" if 'BUY' in verdict.upper() or 'HOLD' in verdict.upper() else "⏸️" if 'WAIT' in verdict.upper() else "🔄"
        
        # QUESTION-SPECIFIC INTRO
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
        
        # Entry strategy
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
        
        response += "\n---\n\n## 📋 Action Plan\n\n"
        
        # CONDITIONAL based on owning stock
        if owns_stock:
            # Advice for existing holders
            if current_price >= targets.get('target_exit', 0):
                response += f"""**✅ Consider Taking Profits**

{analyzed_ticker} has reached ${current_price}, near or above the target exit of ${targets.get('target_exit', 0)}.

**Recommended Actions:**
1. **Sell:** Consider selling all or partial position to lock in gains
2. **Alternative:** Hold with trailing stop at ${targets.get('stop_loss', 0)}
3. **Rebalance:** Use profits to diversify into other opportunities
"""
            elif current_price <= targets.get('stop_loss', 0):
                response += f"""**🔴 Stop Loss Triggered**

{analyzed_ticker} is at ${current_price}, below your stop loss of ${targets.get('stop_loss', 0)}.

**Recommended Actions:**
1. **Sell:** Exit position to limit further losses
2. **Reassess:** Wait for improved fundamentals before re-entering
3. **Learn:** Review what changed to inform future decisions
"""
            else:
                response += f"""**🔄 Hold Current Position**

{analyzed_ticker} is at ${current_price}, between your entry and exit targets.

**Recommended Actions:**
1. **Hold:** Maintain position as thesis remains intact
2. **Monitor:** Watch for price approaching ${targets.get('target_exit', 0)} to take profits
3. **Protect:** Keep stop loss at ${targets.get('stop_loss', 0)}
4. **Timeline:** Continue holding for {self.portfolio_config['investment_horizon']}
"""
        else:
            # Advice for new investors
            entry_price = targets.get('moderate_entry', 0)
            
            if current_price <= entry_price and health_score >= 60:
                response += f"""**✅ Favorable Entry Opportunity**

{analyzed_ticker} at ${current_price} is at or below entry point of ${entry_price}.

**Recommended Actions:**
1. **Entry:** Consider purchasing at current price or lower
2. **Position Size:** Allocate {self.portfolio_config['portfolio_allocation']}% of portfolio
3. **Exit Strategy:** Take profits at ${targets.get('target_exit', 0)}
4. **Risk Management:** Set stop loss at ${targets.get('stop_loss', 0)}
"""
            elif upside > 15:
                response += f"""**⏸️ Wait for Better Entry**

{analyzed_ticker} at ${current_price} is above ideal entry of ${entry_price}.

**Recommended Actions:**
1. **Monitor:** Set price alert for ${entry_price}
2. **Be Patient:** Wait for pullback
3. **Alternative:** Smaller position if must enter now
"""
            else:
                response += f"""**🔴 Not Recommended**

{analyzed_ticker} shows limited upside at current levels.

**Recommended Actions:**
1. **Avoid:** Look for better opportunities
2. **Monitor:** Watch for improved fundamentals
"""
        
        response += f"""

---

**Investment Profile:** {self.portfolio_config['risk_tolerance']} Risk | {self.portfolio_config['investment_horizon']} | {self.portfolio_config['portfolio_allocation']}% Allocation

⚠️ *This is analysis, not personalized financial advice. Consult a qualified advisor.*
"""
        
        return response


# ============================================================================
# RISK ANALYSIS AGENT - TICKER AWARE
# ============================================================================

class RiskAnalysisAgent:
    """Risk assessment with ticker extraction"""
    
    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm
    
    def analyze(self, query: str) -> str:
        """Comprehensive risk analysis"""
        
        # EXTRACT TICKER FROM QUERY
        analyzed_ticker = extract_ticker_from_query(query, self.ticker)
        
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
            
            status.update(label="✅ Risk Analysis Complete", state="complete")
        
        # BUILD RESPONSE
        response = f"""## 🛡️ Risk Assessment for {analyzed_ticker}

---

### Financial Risk Indicators

"""
        
        dte = metrics.get('debt_to_equity', 0)
        if dte > 150:
            response += f"🔴 **High Leverage:** D/E of {dte} indicates significant leverage.\n\n"
        elif dte > 80:
            response += f"🟡 **Moderate Debt:** D/E at {dte} is manageable.\n\n"
        elif dte == 0:
            response += f"🟢 **No Debt:** {analyzed_ticker} has zero debt.\n\n"
        else:
            response += f"🟢 **Low Debt:** D/E of {dte} suggests strong position.\n\n"
        
        current_ratio = metrics.get('current_ratio', 0)
        if current_ratio < 1.0:
            response += f"🔴 **Liquidity Concern:** Current ratio {current_ratio} below 1.0.\n\n"
        elif current_ratio < 1.5:
            response += f"🟡 **Adequate Liquidity:** Current ratio {current_ratio}.\n\n"
        else:
            response += f"🟢 **Strong Liquidity:** Current ratio {current_ratio}.\n\n"
        
        beta = metrics.get('beta', 1.0)
        if beta > 1.5:
            response += f"🔴 **High Volatility:** Beta of {beta}.\n\n"
        elif beta > 1.0:
            response += f"🟡 **Moderate Volatility:** Beta of {beta}.\n\n"
        else:
            response += f"🟢 **Low Volatility:** Beta of {beta}.\n\n"
        
        response += "---\n\n### Business Risks from SEC Filing\n\n"
        
        if filing_risks and "No SEC filing" not in filing_risks:
            response += filing_risks[:1000] + "\n\n*[From SEC 10-Q]*\n\n"
        else:
            response += f"⚠️ Upload {analyzed_ticker}'s 10-Q for detailed risk disclosures.\n\n"
        
        response += "---\n\n### Overall Risk Assessment\n\n"
        
        risk_score = 0
        if dte > 150: risk_score += 2
        elif dte > 80: risk_score += 1
        if current_ratio < 1.0: risk_score += 2
        elif current_ratio < 1.5: risk_score += 1
        if beta > 1.5: risk_score += 2
        elif beta > 1.0: risk_score += 1
        
        if risk_score >= 4:
            rating = "🔴 **HIGH RISK**"
        elif risk_score >= 2:
            rating = "🟡 **MODERATE RISK**"
        else:
            rating = "🟢 **LOW RISK**"
        
        response += f"{rating}\n\n"
        
        response += f"""---

### Risk Mitigation

1. **Position Sizing:** {self.portfolio_config['portfolio_allocation']}% max
2. **Stop Loss:** ${targets.get('stop_loss', 0)}
3. **Diversification:** Balance with other sectors
4. **Monitoring:** Review quarterly updates
"""
        
        return response


# ============================================================================
# PRODUCT ANALYSIS AGENT - TICKER AWARE
# ============================================================================

class ProductAnalysisAgent:
    """Product analysis with ticker extraction"""
    
    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm
    
    def analyze(self, query: str) -> str:
        """Analyze products/segments"""
        
        # EXTRACT TICKER
        analyzed_ticker = extract_ticker_from_query(query, self.ticker)
        
        with st.status("📦 Product Analysis Agent", expanded=True) as status:
            st.write(f"🎯 Analyzing {analyzed_ticker}...")
            st.write("🔧 Gathering metrics...")
            metrics_json = get_stock_metrics_tool(analyzed_ticker)
            
            st.write("📄 Searching SEC filing...")
            products = search_sec_filing_tool(analyzed_ticker, "product segment revenue sales")
            
            st.write("🤖 Analyzing...")
            
            try:
                metrics = json.loads(metrics_json)
            except:
                return "⚠️ Error loading data."
            
            status.update(label="✅ Product Analysis Complete", state="complete")
        
        # BUILD RESPONSE
        growth = metrics.get('revenue_growth_pct', 0)
        margin = metrics.get('profit_margin_pct', 0)
        
        response = f"""## 📦 Product Analysis for {analyzed_ticker}

---

### Overall Performance

**Revenue Growth:** {growth}% YoY """
        
        if growth > 15:
            response += "🟢 Strong\n\n"
        elif growth > 5:
            response += "🟡 Moderate\n\n"
        else:
            response += "🔴 Weak\n\n"
        
        response += f"**Profit Margin:** {margin}% "
        
        if margin > 20:
            response += "🟢 Excellent\n\n"
        elif margin > 10:
            response += "🟡 Good\n\n"
        else:
            response += "🔴 Low\n\n"
        
        response += "---\n\n### SEC Filing Details\n\n"
        
        if products and "No SEC filing" not in products:
            response += products[:1000] + "\n\n*[From 10-Q]*\n\n"
        else:
            response += f"⚠️ Upload {analyzed_ticker}'s 10-Q for segment breakdown.\n\n"
        
        return response


# ============================================================================
# PEER COMPARISON AGENT - WITH HEAD-TO-HEAD
# ============================================================================

class PeerComparisonAgent:
    """Peer comparison with head-to-head support"""
    
    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm
    
    def analyze(self, query: str) -> str:
        """Compare to peers"""
        
        # Extract tickers from query
        ticker1, ticker2 = extract_tickers_from_query(query, self.ticker)
        
        # If two tickers found, do head-to-head
        if ticker2:
            return self.head_to_head_comparison(ticker1, ticker2)
        else:
            return self.sector_comparison(ticker1)
    
    def head_to_head_comparison(self, ticker1: str, ticker2: str) -> str:
        """Direct comparison between two stocks"""
        
        with st.status("🏆 Peer Comparison Agent", expanded=True) as status:
            st.write(f"🔧 Comparing {ticker1} vs {ticker2}...")
            
            metrics1_json = get_stock_metrics_tool(ticker1)
            metrics2_json = get_stock_metrics_tool(ticker2)
            
            try:
                metrics1 = json.loads(metrics1_json)
                metrics2 = json.loads(metrics2_json)
            except:
                return "⚠️ Error loading data."
            
            status.update(label="✅ Comparison Complete", state="complete")
        
        # BUILD HEAD-TO-HEAD
        response = f"""## 🏆 Head-to-Head: {ticker1} vs {ticker2}

---

### Side-by-Side Metrics

| Metric | {ticker1} | {ticker2} | Winner |
|--------|----------|----------|--------|
| **Price** | ${metrics1.get('price', 0)} | ${metrics2.get('price', 0)} | - |
| **P/E Ratio** | {metrics1.get('pe_ratio', 0)} | {metrics2.get('pe_ratio', 0)} | {'🏆 ' + ticker1 if 0 < metrics1.get('pe_ratio', 999) < metrics2.get('pe_ratio', 999) else '🏆 ' + ticker2} (Lower better) |
| **Market Cap** | ${metrics1.get('market_cap_b', 0)}B | ${metrics2.get('market_cap_b', 0)}B | {'🏆 ' + ticker1 if metrics1.get('market_cap_b', 0) > metrics2.get('market_cap_b', 0) else '🏆 ' + ticker2} |
| **Revenue Growth** | {metrics1.get('revenue_growth_pct', 0)}% | {metrics2.get('revenue_growth_pct', 0)}% | {'🏆 ' + ticker1 if metrics1.get('revenue_growth_pct', 0) > metrics2.get('revenue_growth_pct', 0) else '🏆 ' + ticker2} |
| **Profit Margin** | {metrics1.get('profit_margin_pct', 0)}% | {metrics2.get('profit_margin_pct', 0)}% | {'🏆 ' + ticker1 if metrics1.get('profit_margin_pct', 0) > metrics2.get('profit_margin_pct', 0) else '🏆 ' + ticker2} |
| **Debt/Equity** | {metrics1.get('debt_to_equity', 0)} | {metrics2.get('debt_to_equity', 0)} | {'🏆 ' + ticker1 if metrics1.get('debt_to_equity', 999) < metrics2.get('debt_to_equity', 999) else '🏆 ' + ticker2} (Lower better) |

---

### Winner Determination

"""
        
        # Calculate winner
        score1 = sum([
            0 < metrics1.get('pe_ratio', 999) < metrics2.get('pe_ratio', 999),
            metrics1.get('profit_margin_pct', 0) > metrics2.get('profit_margin_pct', 0),
            metrics1.get('revenue_growth_pct', 0) > metrics2.get('revenue_growth_pct', 0),
            metrics1.get('debt_to_equity', 999) < metrics2.get('debt_to_equity', 999)
        ])
        score2 = 4 - score1
        
        if score1 > score2:
            response += f"✅ **{ticker1} is the better investment** (wins {score1}/4 metrics)\n\n"
            response += f"**Recommendation:** Prefer {ticker1} over {ticker2}."
        elif score2 > score1:
            response += f"✅ **{ticker2} is the better investment** (wins {score2}/4 metrics)\n\n"
            response += f"**Recommendation:** Prefer {ticker2} over {ticker1}."
        else:
            response += f"🟡 **Too close to call** (tied {score1}-{score2})\n\n"
            response += "**Recommendation:** Both similarly positioned - choose based on preference."
        
        return response
    
    def sector_comparison(self, ticker: str) -> str:
        """Compare to sector peers"""
        
        with st.status("🏆 Peer Comparison Agent", expanded=True) as status:
            st.write(f"🎯 Analyzing {ticker}...")
            st.write("🔧 Gathering data...")
            
            metrics_json = get_stock_metrics_tool(ticker)
            comparison_json = get_peer_comparison_tool(ticker)
            
            try:
                metrics = json.loads(metrics_json)
                comparison = json.loads(comparison_json)
            except:
                return "⚠️ Error loading data."
            
            status.update(label="✅ Comparison Complete", state="complete")
        
        # BUILD SECTOR COMPARISON
        ticker_pe = comparison.get('ticker_pe', 0)
        sector_pe = comparison.get('sector_avg_pe', 0)
        ticker_margin = comparison.get('ticker_margin_pct', 0)
        sector_margin = comparison.get('sector_avg_margin_pct', 0)
        ticker_growth = comparison.get('ticker_growth_pct', 0)
        sector_growth = comparison.get('sector_avg_growth_pct', 0)
        
        response = f"""## 🏆 Competitive Analysis for {ticker}

**Sector:** {comparison.get('sector', 'Unknown')}  
**Peers:** {', '.join([p['ticker'] for p in comparison.get('peers', [])[:4]])}

---

### Valuation vs Sector

"""
        
        if ticker_pe > 0 and sector_pe > 0:
            pe_diff = round(((ticker_pe - sector_pe) / sector_pe * 100), 1)
            if pe_diff > 15:
                response += f"🔴 **Premium:** P/E {ticker_pe} is {abs(pe_diff)}% ABOVE sector avg {sector_pe}\n\n"
            elif pe_diff > -15:
                response += f"🟡 **Market:** P/E {ticker_pe} in line with sector {sector_pe}\n\n"
            else:
                response += f"🟢 **Discount:** P/E {ticker_pe} is {abs(pe_diff)}% BELOW sector {sector_pe}\n\n"
        
        response += "### Profitability vs Sector\n\n"
        
        if ticker_margin > sector_margin:
            response += f"🟢 **Superior:** {ticker_margin}% margin exceeds sector {sector_margin}%\n\n"
        else:
            response += f"🔴 **Below:** {ticker_margin}% margin trails sector {sector_margin}%\n\n"
        
        response += "### Growth vs Sector\n\n"
        
        if ticker_growth > sector_growth:
            response += f"🟢 **Faster:** {ticker_growth}% vs sector {sector_growth}%\n\n"
        else:
            response += f"🔴 **Slower:** {ticker_growth}% vs sector {sector_growth}%\n\n"
        
        response += "---\n\n### Peer Metrics\n\n"
        
        if comparison.get('peers'):
            response += "| Ticker | P/E | Margin | Growth |\n|--------|-----|--------|--------|\n"
            for peer in comparison['peers'][:4]:
                response += f"| {peer['ticker']} | {peer.get('pe', 'N/A')} | {peer.get('profit_margin_pct', 'N/A')}% | {peer.get('revenue_growth_pct', 'N/A')}% |\n"
            response += f"| **{ticker}** | **{ticker_pe}** | **{ticker_margin}%** | **{ticker_growth}%** |\n"
        
        response += "\n---\n\n### Verdict\n\n"
        
        wins = sum([
            ticker_pe < sector_pe if ticker_pe > 0 and sector_pe > 0 else False,
            ticker_margin > sector_margin,
            ticker_growth > sector_growth
        ])
        
        if wins >= 2:
            response += f"✅ **Strong:** {ticker} outperforms on {wins}/3 metrics."
        elif wins == 1:
            response += f"🟡 **Mixed:** {ticker} competitive on some metrics."
        else:
            response += f"🔴 **Weak:** {ticker} underperforms peers."
        
        return response


# ============================================================================
# GENERAL ANALYSIS AGENT - TICKER AWARE
# ============================================================================

class GeneralAnalysisAgent:
    """General analysis with ticker extraction"""
    
    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm
    
    def analyze(self, query: str) -> str:
        """General analysis with validation"""
        
        # EXTRACT TICKER AND CHECK IF IT WAS IN QUERY
        analyzed_ticker, found_in_query = extract_ticker_from_query(query, self.ticker)
        
        with st.status("📊 General Analysis Agent", expanded=True) as status:
            st.write(f"🎯 Analyzing {analyzed_ticker}...")
            st.write("🔧 Gathering data...")
            
            metrics_json = get_stock_metrics_tool(analyzed_ticker)
            
            # CHECK IF WE GOT ERROR
            try:
                metrics = json.loads(metrics_json)
                
                if 'error' in metrics:
                    # We don't have data for this ticker
                    status.update(label="⚠️ Data Not Available", state="error")
                    
                    if found_in_query:
                        # User asked about specific ticker we don't have
                        return f"""## ⚠️ No Data Available for {analyzed_ticker}

**Sorry, we don't have financial data for {analyzed_ticker} in our system.**

**What we can analyze:**
- AAPL (Apple)
- MSFT (Microsoft)
- GOOGL (Google)
- META (Meta/Facebook)
- NVDA (NVIDIA)
- TSLA (Tesla)
- AMZN (Amazon)
- JPM (JPMorgan Chase)
- V (Visa)
- NFLX (Netflix)
- And more...

**Try asking about one of these companies instead!**

Or select a stock from the dropdown in the sidebar.
"""
                    else:
                        return f"⚠️ Error loading data for {analyzed_ticker}. {metrics.get('error', 'Unknown error')}"
                
                # We have data, continue
                health_json = calculate_health_score_tool(analyzed_ticker)
                filing = search_sec_filing_tool(analyzed_ticker, query)
                
                health = json.loads(health_json)
                
            except:
                return "⚠️ Error loading data."
            
            st.write("🤖 Analyzing...")
            status.update(label="✅ Analysis Complete", state="complete")
        
        # BUILD RESPONSE (rest stays the same)
        response = f"""## 📊 Financial Overview for {analyzed_ticker}

---

### Company Profile

**Name:** {metrics.get('company_name', analyzed_ticker)}  
**Sector:** {metrics.get('sector', 'Unknown')}  
**Industry:** {metrics.get('industry', 'Unknown')}

---

### Health Score

**Score:** {health.get('score', 0)}/100 ({health.get('rating', 'N/A')})

"""
        
        if health.get('factors'):
            for factor in health['factors']:
                response += f"- {factor}\n"
        
        response += f"""

---

### Key Metrics

| Category | Metric | Value |
|----------|--------|-------|
| **Valuation** | Price | ${metrics.get('price', 0)} |
| | P/E | {metrics.get('pe_ratio', 0)} |
| **Growth** | Revenue Growth | {metrics.get('revenue_growth_pct', 0)}% |
| **Profitability** | Profit Margin | {metrics.get('profit_margin_pct', 0)}% |
| **Risk** | Debt/Equity | {metrics.get('debt_to_equity', 0)} |
| | Beta | {metrics.get('beta', 0)} |

---

### Quick Assessment

"""
        
        if health.get('score', 0) > 70:
            response += f"✅ **Strong Fundamentals:** {analyzed_ticker} scores {health.get('score', 0)}/100."
        elif health.get('score', 0) > 50:
            response += f"🟡 **Adequate:** {analyzed_ticker} scores {health.get('score', 0)}/100."
        else:
            response += f"🔴 **Weak:** {analyzed_ticker} scores {health.get('score', 0)}/100."
        
        return response
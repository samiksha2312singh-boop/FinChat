"""
Multi-Agent System using Fireworks Llama 3.1 70B
Each agent specializes in a specific type of financial analysis
REVISED: Question-aware responses, proper variation
"""

import streamlit as st
from langchain_fireworks import ChatFireworks
from langchain.prompts import ChatPromptTemplate
from typing import Dict
import json
from utils.tools import (
    get_stock_metrics_tool,
    search_sec_filing_tool,
    calculate_price_targets_tool,
    get_peer_comparison_tool,
    calculate_health_score_tool
)

def extract_tickers_from_query(query: str, current_ticker: str) -> tuple:
    """Extract ticker symbols mentioned in the query"""
    
    all_tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMZN', 
                   'JPM', 'V', 'JNJ', 'WMT', 'PG', 'NFLX', 'DIS', 'GM', 'F', 'RIVN']
    
    query_upper = query.upper()
    found_tickers = [t for t in all_tickers if t in query_upper]
    
    if len(found_tickers) >= 2:
        return found_tickers[0], found_tickers[1]
    elif len(found_tickers) == 1:
        return found_tickers[0], None
    else:
        return current_ticker, None

@st.cache_resource
def get_llm():
    """Initialize Fireworks Llama 3.1 70B"""
    llm = ChatFireworks(
        model="accounts/fireworks/models/llama-v3p3-70b-instruct",
        api_key=st.secrets["FIREWORKS_API_KEY"],
        temperature=0.7,  # Increased for more natural variation
        max_tokens=1024
    )
    return llm


class FinancialAgentOrchestrator:
    """Routes queries to specialized agents"""
    
    def __init__(self, ticker: str, portfolio_config: Dict):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = get_llm()
    
    def classify_intent(self, query: str) -> str:
        """Classify user query"""
        query_lower = query.lower()
        
        investment_keywords = ['invest', 'buy', 'sell', 'hold', 'price target', 'should i', 'entry', 'exit', 'good buy']
        risk_keywords = ['risk', 'danger', 'concern', 'threat', 'safe', 'risky', 'worry', 'worried']
        product_keywords = ['product', 'segment', 'business', 'revenue', 'sales', 'performing', 'advertising']
        comparison_keywords = ['compare', 'competitor', 'peer', 'vs', 'versus', 'better than']
        
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


class InvestmentAdvisorAgent:
    """Investment decisions with question-aware responses"""
    
    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm
    
    def analyze(self, query: str) -> str:
        """Provide investment recommendation tailored to specific question"""
        
        with st.status("💼 Investment Advisor Agent", expanded=True) as status:
            st.write("🔧 Gathering financial metrics...")
            metrics_json = get_stock_metrics_tool(self.ticker)
            
            st.write("🎯 Calculating price targets...")
            targets_json = calculate_price_targets_tool(self.ticker, self.portfolio_config['risk_tolerance'])
            
            st.write("🏥 Assessing financial health...")
            health_json = calculate_health_score_tool(self.ticker)
            
            st.write("📄 Checking SEC filing...")
            filing_risks = search_sec_filing_tool(self.ticker, "risks concerns")
            
            st.write("🤖 Generating recommendation...")
            
            # Parse data
            try:
                metrics = json.loads(metrics_json)
                targets = json.loads(targets_json)
                health = json.loads(health_json)
            except:
                return "⚠️ Error parsing financial data. Please try again."
            
            # QUESTION-AWARE VERDICT - Different prompts for different questions
            query_lower = query.lower()
            
            if 'conservative' in query_lower:
                verdict_prompt = f"""You are advising a CONSERVATIVE investor about {self.ticker}.

Current Price: ${targets.get('current_price', 0)}
Conservative Entry: ${targets.get('conservative_entry', 0)}
Health Score: {health.get('score', 0)}/100
Debt/Equity: {metrics.get('debt_to_equity', 0)}
Beta: {metrics.get('beta', 0)}

Question: {query}

Focus on safety and downside protection. Provide: "[BUY/HOLD/WAIT] - [emphasize risk factors and safety]"
"""
            elif 'entry price' in query_lower or 'entry point' in query_lower:
                verdict_prompt = f"""You are advising on the BEST ENTRY PRICE for {self.ticker}.

Current Price: ${targets.get('current_price', 0)}
Fair Value: ${targets.get('fair_value', 0)}
Moderate Entry: ${targets.get('moderate_entry', 0)}
Conservative Entry: ${targets.get('conservative_entry', 0)}

Question: {query}

Focus on optimal entry timing and price levels. Provide: "[BUY/HOLD/WAIT] - [emphasize entry price strategy]"
"""
            else:
                verdict_prompt = f"""You are a financial advisor answering: "{query}"

Data for {self.ticker}:
Current Price: ${targets.get('current_price', 0)}
Fair Value: ${targets.get('fair_value', 0)}
Upside: {targets.get('upside_percent', 0)}%
Health Score: {health.get('score', 0)}/100
Revenue Growth: {metrics.get('revenue_growth_pct', 0)}%
Profit Margin: {metrics.get('profit_margin_pct', 0)}%

Answer SPECIFICALLY: "{query}"

Provide: "[BUY/HOLD/WAIT] - [tailor explanation to their specific question]"
"""
            
            verdict_response = self.llm.invoke(verdict_prompt)
            verdict = verdict_response.content.strip()
            
            status.update(label="✅ Investment Advisor Complete", state="complete")
        
        # BUILD CLEAN RESPONSE PROGRAMMATICALLY
        current_price = targets.get('current_price', 0)
        fair_value = targets.get('fair_value', 0)
        upside = targets.get('upside_percent', 0)
        health_score = health.get('score', 0)
        health_rating = health.get('rating', 'N/A')
        
        # Determine recommendation emoji
        if 'BUY' in verdict.upper():
            verdict_emoji = "✅"
        elif 'WAIT' in verdict.upper():
            verdict_emoji = "⏸️"
        else:
            verdict_emoji = "🔄"
        
        # QUESTION-SPECIFIC INTRO
        query_lower = query.lower()
        if 'conservative' in query_lower:
            intro = f"**For Conservative Investors:**\n\n"
        elif 'aggressive' in query_lower:
            intro = f"**For Aggressive Investors:**\n\n"
        elif 'entry' in query_lower:
            intro = f"**Entry Price Analysis:**\n\n"
        else:
            intro = ""
        
        response = f"""## {verdict_emoji} Investment Recommendation for {self.ticker}

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
        
        # HIGHLIGHT based on risk tolerance
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
        else:  # Moderate
            response += f"""- **🎯 Moderate Entry:** ${targets.get('moderate_entry', 0)} **(recommended for you)**
- Conservative Entry: ${targets.get('conservative_entry', 0)} (safer)
- Aggressive Entry: ${targets.get('aggressive_entry', 0)} (higher risk)
"""
        
        response += f"""- **Target Exit:** ${targets.get('target_exit', 0)} (+{round((targets.get('target_exit', 0) / current_price - 1) * 100, 1) if current_price > 0 else 0}%)
- **Stop Loss:** ${targets.get('stop_loss', 0)} (-8% protection)

---

## 📈 Key Financial Metrics

| Metric | {self.ticker} | Assessment |
|--------|------|------------|
| **Revenue Growth** | {metrics.get('revenue_growth_pct', 0)}% | {'🟢 Strong' if metrics.get('revenue_growth_pct', 0) > 15 else '🟡 Moderate' if metrics.get('revenue_growth_pct', 0) > 5 else '🔴 Weak'} |
| **Profit Margin** | {metrics.get('profit_margin_pct', 0)}% | {'🟢 Excellent' if metrics.get('profit_margin_pct', 0) > 20 else '🟡 Good' if metrics.get('profit_margin_pct', 0) > 10 else '🔴 Low'} |
| **Debt/Equity** | {metrics.get('debt_to_equity', 0)} | {'🟢 Low Risk' if metrics.get('debt_to_equity', 0) < 50 else '🟡 Moderate' if metrics.get('debt_to_equity', 0) < 150 else '🔴 High Risk'} |
| **Beta (Volatility)** | {metrics.get('beta', 0)} | {'🟢 Below Market' if metrics.get('beta', 0) < 1.0 else '🟡 Market Average' if metrics.get('beta', 0) < 1.3 else '🔴 High Volatility'} |

---

## ⚠️ Key Considerations

"""
        
        # Add health factors
        if health.get('factors'):
            for i, factor in enumerate(health['factors'][:3], 1):
                response += f"{i}. {factor}\n"
        
        response += "\n---\n\n## 📋 Action Plan\n\n"
        
        # CONDITIONAL ACTION PLAN
        entry_price = targets.get('moderate_entry', 0)
        
        if current_price <= entry_price and health_score >= 60:
            response += f"""**✅ Favorable Entry Opportunity**

The current price of ${current_price} is at or below the recommended entry point of ${entry_price}, with a strong health score of {health_score}/100.

**Recommended Actions:**
1. **Entry:** Consider purchasing at current price or lower
2. **Position Size:** Allocate up to {self.portfolio_config['portfolio_allocation']}% of your portfolio
3. **Exit Strategy:** Take profits at ${targets.get('target_exit', 0)}
4. **Risk Management:** Set stop loss at ${targets.get('stop_loss', 0)}
5. **Timeline:** Hold for {self.portfolio_config['investment_horizon']} based on your horizon
"""
        elif upside > 15 and health_score >= 50:
            response += f"""**⏸️ Wait for Better Entry**

While {self.ticker} shows {upside}% upside potential, the current price of ${current_price} is above the ideal entry point of ${entry_price}.

**Recommended Actions:**
1. **Monitor:** Set price alert for ${entry_price}
2. **Be Patient:** Wait for pullback before entering
3. **Alternative:** If must enter now, use smaller position ({round(self.portfolio_config['portfolio_allocation'] / 2, 1)}%)
4. **Review:** Reassess if price falls to entry zone
"""
        else:
            response += f"""**🔴 Not Recommended at Current Levels**

With a health score of {health_score}/100 and limited upside of {upside}%, {self.ticker} does not present a compelling opportunity at ${current_price}.

**Recommended Actions:**
1. **Avoid:** Look for better opportunities elsewhere
2. **Monitor:** Watch for significant improvement in fundamentals
3. **Alternatives:** Consider other stocks in the {metrics.get('sector', 'same')} sector
4. **Reassess:** Revisit if metrics improve significantly
"""
        
        # QUESTION-SPECIFIC ADDENDUM
        if 'conservative' in query_lower:
            response += f"""

---

### 🛡️ Note for Conservative Investors

For your conservative profile, consider:
- **Entry only at:** ${targets.get('conservative_entry', 0)} (15% below fair value)
- **Smaller position:** {round(self.portfolio_config['portfolio_allocation'] * 0.7, 1)}% allocation
- **Stricter stop:** ${round(current_price * 0.95, 2)} (-5% instead of -8%)
- **Beta consideration:** {metrics.get('beta', 0)} volatility {'may be too high' if metrics.get('beta', 0) > 1.2 else 'is acceptable'}
"""
        
        response += f"""

---

**Investment Profile:** {self.portfolio_config['risk_tolerance']} Risk | {self.portfolio_config['investment_horizon']} Horizon | {self.portfolio_config['portfolio_allocation']}% Allocation

⚠️ *This is analysis, not personalized financial advice. Consult a qualified advisor.*
"""
        
        return response


class RiskAnalysisAgent:
    """Risk assessment with SEC filing integration"""
    
    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm
    
    def analyze(self, query: str) -> str:
        """Comprehensive risk analysis"""
        
        with st.status("🛡️ Risk Analysis Agent", expanded=True) as status:
            st.write("🔧 Gathering financial metrics...")
            metrics_json = get_stock_metrics_tool(self.ticker)
            
            st.write("🎯 Calculating risk thresholds...")
            targets_json = calculate_price_targets_tool(self.ticker, self.portfolio_config['risk_tolerance'])  # ADD THIS LINE
            
            st.write("📄 Extracting SEC filing risks...")
            filing_risks = search_sec_filing_tool(self.ticker, "risk factors threats concerns regulatory operational financial")
            
            st.write("🤖 Analyzing risk profile...")
            
            # Parse metrics AND targets
            try:
                metrics = json.loads(metrics_json)
                targets = json.loads(targets_json)  # ADD THIS LINE
            except:
                return "⚠️ Error loading metrics data."
            
            status.update(label="✅ Risk Analysis Complete", state="complete")
        
        # BUILD RESPONSE
        response = f"""## 🛡️ Risk Assessment for {self.ticker}

---

### Financial Risk Indicators

"""
        
        # Debt risk
        dte = metrics.get('debt_to_equity', 0)
        if dte > 150:
            response += f"🔴 **High Leverage Risk:** Debt-to-Equity ratio of {dte} indicates significant leverage.\n\n"
        elif dte > 80:
            response += f"🟡 **Moderate Debt Levels:** Debt-to-Equity at {dte} is manageable but warrants monitoring.\n\n"
        elif dte == 0:
            response += f"🟢 **No Debt:** {self.ticker} operates with zero debt, providing maximum financial flexibility.\n\n"
        else:
            response += f"🟢 **Low Debt Risk:** Conservative D/E ratio of {dte} suggests strong financial position.\n\n"
        
        # Liquidity risk
        current_ratio = metrics.get('current_ratio', 0)
        if current_ratio < 1.0:
            response += f"🔴 **Liquidity Concern:** Current ratio of {current_ratio} is below 1.0, indicating potential short-term payment issues.\n\n"
        elif current_ratio < 1.5:
            response += f"🟡 **Adequate Liquidity:** Current ratio of {current_ratio} suggests sufficient short-term assets.\n\n"
        else:
            response += f"🟢 **Strong Liquidity:** Current ratio of {current_ratio} indicates excellent ability to cover short-term obligations.\n\n"
        
        # Market risk
        beta = metrics.get('beta', 1.0)
        if beta > 1.5:
            response += f"🔴 **High Market Risk:** Beta of {beta} indicates high volatility relative to market.\n\n"
        elif beta > 1.0:
            response += f"🟡 **Moderate Volatility:** Beta of {beta} suggests slightly higher volatility than market average.\n\n"
        else:
            response += f"🟢 **Low Volatility:** Beta of {beta} indicates lower risk than broad market.\n\n"
        
        response += "---\n\n### Business Risks from SEC Filing\n\n"
        
        # Add filing risks if available
        if filing_risks and "No SEC filing" not in filing_risks and "error" not in filing_risks.lower():
            response += filing_risks[:1000] + "\n\n"
            response += "*[Extracted from uploaded SEC 10-Q filing]*\n\n"
        else:
            response += f"""⚠️ **No detailed SEC filing data available for {self.ticker}**

Upload the latest 10-Q to see:
- Specific risk factors disclosed by management
- Regulatory and compliance risks
- Operational challenges
- Market and competitive threats

For now, risk assessment is based on financial metrics only.

"""
        
        response += "---\n\n### Overall Risk Assessment\n\n"
        
        # Calculate overall risk score
        risk_score = 0
        if dte > 150: risk_score += 2
        elif dte > 80: risk_score += 1
        
        if current_ratio < 1.0: risk_score += 2
        elif current_ratio < 1.5: risk_score += 1
        
        if beta > 1.5: risk_score += 2
        elif beta > 1.0: risk_score += 1
        
        if risk_score >= 4:
            rating = "🔴 **HIGH RISK**"
            explanation = "Multiple concerning financial indicators suggest elevated investment risk. Suitable only for aggressive risk tolerance."
        elif risk_score >= 2:
            rating = "🟡 **MODERATE RISK**"
            explanation = "Some risk factors present. Appropriate for moderate risk tolerance investors with proper position sizing."
        else:
            rating = "🟢 **LOW RISK**"
            explanation = "Strong financial metrics suggest relatively low investment risk. Suitable for most risk profiles."
        
        response += f"{rating}\n\n{explanation}\n\n"
        
        response += f"""---

### Risk Mitigation Strategies

1. **Position Sizing:** Limit to {self.portfolio_config['portfolio_allocation']}% of portfolio to manage exposure
2. **Stop Loss:** Set at ${targets.get('stop_loss', 0)} to limit downside to -8%
3. **Diversification:** Balance with {'lower-beta stocks' if beta > 1.2 else 'other sectors'} to reduce portfolio volatility
4. **Monitoring:** Review quarterly earnings and SEC filing updates for emerging risks
5. **Horizon Alignment:** {self.portfolio_config['investment_horizon']} hold allows riding through volatility

---

*Risk assessment for {self.portfolio_config['risk_tolerance']} tolerance investor.*
"""
        
        return response

class ProductAnalysisAgent:
    """Product/segment analysis with SEC filing extraction"""
    
    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm
    
    def analyze(self, query: str) -> str:
        """Analyze products/segments"""
        
        with st.status("📦 Product Analysis Agent", expanded=True) as status:
            st.write("🔧 Gathering metrics...")
            metrics_json = get_stock_metrics_tool(self.ticker)
            
            st.write("📄 Searching SEC filing for product data...")
            products = search_sec_filing_tool(self.ticker, "product segment revenue sales performance growth business")
            
            st.write("🤖 Analyzing...")
            
            try:
                metrics = json.loads(metrics_json)
            except:
                return "⚠️ Error loading data."
            
            status.update(label="✅ Product Analysis Complete", state="complete")
        
        # BUILD RESPONSE
        response = f"""## 📦 Product & Segment Analysis for {self.ticker}

---

### Overall Company Performance

"""
        
        growth = metrics.get('revenue_growth_pct', 0)
        margin = metrics.get('profit_margin_pct', 0)
        
        response += f"""**Revenue Growth:** {growth}% YoY """
        if growth > 15:
            response += "🟢 Strong growth indicates successful product portfolio with multiple drivers\n\n"
        elif growth > 5:
            response += "🟡 Moderate growth suggests stable but mature product mix\n\n"
        else:
            response += "🔴 Weak growth may indicate product challenges or market saturation\n\n"
        
        response += f"""**Profit Margin:** {margin}% """
        if margin > 20:
            response += "🟢 Premium margins suggest strong product differentiation and pricing power\n\n"
        elif margin > 10:
            response += "🟡 Healthy margins but facing some competitive pressure\n\n"
        else:
            response += "🔴 Thin margins indicate commoditization risk or cost challenges\n\n"
        
        response += "---\n\n### Product/Segment Details from SEC Filing\n\n"
        
        # Add filing data
        if products and "No SEC filing" not in products and "error" not in products.lower():
            response += products[:1200] + "\n\n"
            response += "*[Extracted from uploaded SEC 10-Q filing]*\n\n"
        else:
            response += f"""⚠️ **Limited product segment data available**

To get detailed product/segment breakdown:
- Upload {self.ticker}'s latest 10-Q filing
- The MD&A section contains revenue by product category
- Segment performance with growth rates
- Geographic breakdowns

**Based on overall metrics:**
- Product portfolio appears {"strong and competitive" if growth > 10 and margin > 15 else "stable but mature" if growth > 0 else "challenged"}
- {"Multiple growth drivers likely" if growth > 15 else "Steady performance" if growth > 5 else "May need new products or markets"}

"""
        
        response += f"""---

### Investment Implications

**Product Health Impact:**
- Revenue growth of {growth}% {'supports' if growth > 10 else 'moderately supports' if growth > 0 else 'raises concerns about'} investment thesis
- {margin}% margins {'indicate strong competitive position' if margin > 20 else 'are adequate' if margin > 10 else 'suggest pricing pressure'}

**Recommendation:** Upload SEC filing for deeper product analysis before making investment decision.
"""
        
        return response


class PeerComparisonAgent:
    """Peer comparison with detailed tables"""
    
    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm
    
    def analyze(self, query: str) -> str:
        """Compare to peers"""
        
        # Extract tickers from query
        ticker1, ticker2 = extract_tickers_from_query(query, self.ticker)
        
        # If comparing two specific stocks (e.g., "Is META better than GOOGL?")
        if ticker2:
            return self.head_to_head_comparison(ticker1, ticker2, query)
        else:
            # Sector comparison for single ticker
            return self.sector_comparison(ticker1, query)
    
    def head_to_head_comparison(self, ticker1: str, ticker2: str, query: str) -> str:
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
        
        # Build head-to-head response
        response = f"""## 🏆 Head-to-Head: {ticker1} vs {ticker2}

---

### Side-by-Side Metrics

| Metric | {ticker1} | {ticker2} | Winner |
|--------|----------|----------|--------|
| **Price** | ${metrics1.get('price', 0)} | ${metrics2.get('price', 0)} | - |
| **P/E Ratio** | {metrics1.get('pe_ratio', 0)} | {metrics2.get('pe_ratio', 0)} | {'🏆 ' + ticker1 if 0 < metrics1.get('pe_ratio', 999) < metrics2.get('pe_ratio', 999) else '🏆 ' + ticker2} |
| **Market Cap** | ${metrics1.get('market_cap_b', 0)}B | ${metrics2.get('market_cap_b', 0)}B | {'🏆 ' + ticker1 if metrics1.get('market_cap_b', 0) > metrics2.get('market_cap_b', 0) else '🏆 ' + ticker2} |
| **Revenue Growth** | {metrics1.get('revenue_growth_pct', 0)}% | {metrics2.get('revenue_growth_pct', 0)}% | {'🏆 ' + ticker1 if metrics1.get('revenue_growth_pct', 0) > metrics2.get('revenue_growth_pct', 0) else '🏆 ' + ticker2} |
| **Profit Margin** | {metrics1.get('profit_margin_pct', 0)}% | {metrics2.get('profit_margin_pct', 0)}% | {'🏆 ' + ticker1 if metrics1.get('profit_margin_pct', 0) > metrics2.get('profit_margin_pct', 0) else '🏆 ' + ticker2} |
| **Debt/Equity** | {metrics1.get('debt_to_equity', 0)} | {metrics2.get('debt_to_equity', 0)} | {'🏆 ' + ticker1 if metrics1.get('debt_to_equity', 999) < metrics2.get('debt_to_equity', 999) else '🏆 ' + ticker2} |

---

### Overall Winner

"""
        
        # Calculate winner
        score1 = 0
        score2 = 0
        
        if 0 < metrics1.get('pe_ratio', 999) < metrics2.get('pe_ratio', 999): score1 += 1
        else: score2 += 1
        
        if metrics1.get('profit_margin_pct', 0) > metrics2.get('profit_margin_pct', 0): score1 += 1
        else: score2 += 1
        
        if metrics1.get('revenue_growth_pct', 0) > metrics2.get('revenue_growth_pct', 0): score1 += 1
        else: score2 += 1
        
        if metrics1.get('debt_to_equity', 999) < metrics2.get('debt_to_equity', 999): score1 += 1
        else: score2 += 1
        
        if score1 > score2:
            response += f"✅ **{ticker1} is the better investment** (wins {score1}/4 metrics)\n\n"
        elif score2 > score1:
            response += f"✅ **{ticker2} is the better investment** (wins {score2}/4 metrics)\n\n"
        else:
            response += f"🟡 **Too close to call** (tied at {score1}-{score2})\n\n"
        
        response += f"**Recommendation:** Based on fundamentals, "
        if score1 > score2:
            response += f"prefer {ticker1} over {ticker2}."
        elif score2 > score1:
            response += f"prefer {ticker2} over {ticker1}."
        else:
            response += "both are similarly positioned - choose based on sector preference."
        
        return response
    
    
    def sector_comparison(self, ticker: str, query: str) -> str:
        """Compare to sector peers (original logic)"""
        
        with st.status("🏆 Peer Comparison Agent", expanded=True) as status:
            st.write(f"🔧 Gathering data for {ticker}...")
            
            metrics_json = get_stock_metrics_tool(ticker)
            comparison_json = get_peer_comparison_tool(ticker)
            
            try:
                metrics = json.loads(metrics_json)
                comparison = json.loads(comparison_json)
            except:
                return "⚠️ Error loading data."
            
            status.update(label="✅ Comparison Complete", state="complete")
        
        # [REST OF YOUR CURRENT SECTOR COMPARISON CODE]
        # Copy the entire sector comparison logic you already have
        ticker_pe = comparison.get('ticker_pe', 0)
        sector_pe = comparison.get('sector_avg_pe', 0)
        ticker_margin = comparison.get('ticker_margin_pct', 0)
        sector_margin = comparison.get('sector_avg_margin_pct', 0)
        ticker_growth = comparison.get('ticker_growth_pct', 0)
        sector_growth = comparison.get('sector_avg_growth_pct', 0)
        
        response = f"""## 🏆 Competitive Analysis for {ticker}

**Sector:** {comparison.get('sector', 'Unknown')}  
**Comparing to:** {', '.join([p['ticker'] for p in comparison.get('peers', [])[:4]])}

---

### Valuation Comparison

"""
        
        if ticker_pe > 0 and sector_pe > 0:
            pe_diff = round(((ticker_pe - sector_pe) / sector_pe * 100), 1)
            if pe_diff > 15:
                response += f"🔴 **Premium Valuation:** {ticker}'s P/E of {ticker_pe} is {abs(pe_diff)}% ABOVE sector average of {sector_pe}\n\n"
            elif pe_diff > -15:
                response += f"🟡 **Market Valuation:** {ticker}'s P/E of {ticker_pe} is roughly in line with sector average of {sector_pe}\n\n"
            else:
                response += f"🟢 **Discount Valuation:** {ticker}'s P/E of {ticker_pe} is {abs(pe_diff)}% BELOW sector average of {sector_pe}\n\n"
        
        response += "### Profitability Comparison\n\n"
        
        if ticker_margin > sector_margin:
            margin_diff = round(ticker_margin - sector_margin, 1)
            response += f"🟢 **Above Average:** {ticker}'s {ticker_margin}% margin exceeds sector of {sector_margin}% by {margin_diff} pts\n\n"
        else:
            margin_diff = round(sector_margin - ticker_margin, 1)
            response += f"🔴 **Below Average:** {ticker}'s {ticker_margin}% margin trails sector of {sector_margin}% by {margin_diff} pts\n\n"
        
        response += "### Growth Comparison\n\n"
        
        if ticker_growth > sector_growth:
            response += f"🟢 **Faster Growth:** {ticker} at {ticker_growth}% vs sector {sector_growth}%\n\n"
        else:
            response += f"🔴 **Slower Growth:** {ticker} at {ticker_growth}% vs sector {sector_growth}%\n\n"
        
        response += "---\n\n### Detailed Peer Metrics\n\n"
        
        if comparison.get('peers'):
            response += "| Ticker | P/E | Profit Margin | Revenue Growth |\n|--------|-----|---------------|----------------|\n"
            for peer in comparison['peers'][:4]:
                response += f"| {peer['ticker']} | {peer.get('pe', 'N/A')} | {peer.get('profit_margin_pct', 'N/A')}% | {peer.get('revenue_growth_pct', 'N/A')}% |\n"
            response += f"| **{ticker}** | **{ticker_pe}** | **{ticker_margin}%** | **{ticker_growth}%** |\n"
            response += f"| *Sector Avg* | *{sector_pe}* | *{sector_margin}%* | *{sector_growth}%* |\n\n"
        
        response += "---\n\n### Investment Verdict\n\n"
        
        beats_valuation = ticker_pe < sector_pe if ticker_pe > 0 and sector_pe > 0 else False
        beats_profitability = ticker_margin > sector_margin
        beats_growth = ticker_growth > sector_growth
        wins = sum([beats_valuation, beats_profitability, beats_growth])
        
        if wins >= 2:
            response += f"✅ **Strong Position:** {ticker} outperforms on {wins}/3 metrics."
        elif wins == 1:
            response += f"🟡 **Mixed Position:** {ticker} competitive on some metrics."
        else:
            response += f"🔴 **Weak Position:** {ticker} underperforms peers."
        
        return response
        
        # BUILD RESPONSE
        ticker_pe = comparison.get('ticker_pe', 0)
        sector_pe = comparison.get('sector_avg_pe', 0)
        ticker_margin = comparison.get('ticker_margin_pct', 0)
        sector_margin = comparison.get('sector_avg_margin_pct', 0)
        ticker_growth = comparison.get('ticker_growth_pct', 0)
        sector_growth = comparison.get('sector_avg_growth_pct', 0)
        
        response = f"""## 🏆 Competitive Analysis for {self.ticker}

**Sector:** {comparison.get('sector', 'Unknown')}  
**Comparing to:** {', '.join([p['ticker'] for p in comparison.get('peers', [])[:4]])}

---

### Valuation Comparison

"""
        
        if ticker_pe > 0 and sector_pe > 0:
            pe_diff = round(((ticker_pe - sector_pe) / sector_pe * 100), 1)
            if pe_diff > 15:
                response += f"🔴 **Premium Valuation:** {self.ticker}'s P/E of {ticker_pe} is {abs(pe_diff)}% ABOVE sector average of {sector_pe}\n\n"
                response += f"*Implication: Investors are paying a premium, expecting superior growth or quality.*\n\n"
            elif pe_diff > -15:
                response += f"🟡 **Market Valuation:** {self.ticker}'s P/E of {ticker_pe} is roughly in line with sector average of {sector_pe}\n\n"
                response += f"*Implication: Fair market pricing relative to peers.*\n\n"
            else:
                response += f"🟢 **Discount Valuation:** {self.ticker}'s P/E of {ticker_pe} is {abs(pe_diff)}% BELOW sector average of {sector_pe}\n\n"
                response += f"*Implication: Potential value opportunity if fundamentals are solid.*\n\n"
        
        response += "### Profitability Comparison\n\n"
        
        if ticker_margin > sector_margin:
            margin_diff = round(ticker_margin - sector_margin, 1)
            response += f"🟢 **Superior Profitability:** {self.ticker}'s {ticker_margin}% margin EXCEEDS sector average of {sector_margin}% by {margin_diff} percentage points\n\n"
            response += f"*This indicates better operational efficiency or pricing power than competitors.*\n\n"
        else:
            margin_diff = round(sector_margin - ticker_margin, 1)
            response += f"🔴 **Below Sector:** {self.ticker}'s {ticker_margin}% margin TRAILS sector average of {sector_margin}% by {margin_diff} percentage points\n\n"
            response += f"*May face cost pressures or weaker competitive position.*\n\n"
        
        response += "### Growth Comparison\n\n"
        
        if ticker_growth > sector_growth:
            growth_diff = round(ticker_growth - sector_growth, 1)
            response += f"🟢 **Faster Growth:** {self.ticker} growing at {ticker_growth}% vs sector average of {sector_growth}% (+{growth_diff} pts)\n\n"
            response += f"*Outgrowing competitors suggests market share gains or expansion success.*\n\n"
        else:
            growth_diff = round(sector_growth - ticker_growth, 1)
            response += f"🔴 **Slower Growth:** {self.ticker} growing at {ticker_growth}% vs sector average of {sector_growth}% (-{growth_diff} pts)\n\n"
            response += f"*May be losing market share or facing headwinds.*\n\n"
        
        response += "---\n\n### Detailed Peer Metrics\n\n"
        
        # Peer table
        if comparison.get('peers'):
            response += "| Ticker | P/E Ratio | Profit Margin | Revenue Growth |\n"
            response += "|--------|-----------|---------------|----------------|\n"
            for peer in comparison['peers'][:4]:
                response += f"| {peer['ticker']} | {peer.get('pe', 'N/A')} | {peer.get('profit_margin_pct', 'N/A')}% | {peer.get('revenue_growth_pct', 'N/A')}% |\n"
            response += f"| **{self.ticker}** | **{ticker_pe}** | **{ticker_margin}%** | **{ticker_growth}%** |\n"
            response += f"| *Sector Avg* | *{sector_pe}* | *{sector_margin}%* | *{sector_growth}%* |\n\n"
        
        response += "---\n\n### Competitive Scorecard\n\n"
        
        # Determine wins
        beats_valuation = ticker_pe < sector_pe if ticker_pe > 0 and sector_pe > 0 else False
        beats_profitability = ticker_margin > sector_margin
        beats_growth = ticker_growth > sector_growth
        
        wins = sum([beats_valuation, beats_profitability, beats_growth])
        
        response += f"""**Metrics Won:** {wins}/3

- {'✅' if beats_valuation else '❌'} Valuation (P/E): {'Cheaper' if beats_valuation else 'More expensive'} than peers
- {'✅' if beats_profitability else '❌'} Profitability: {'Higher' if beats_profitability else 'Lower'} margins than peers
- {'✅' if beats_growth else '❌'} Growth: {'Faster' if beats_growth else 'Slower'} than peers

---

### Investment Verdict

"""
        
        if wins >= 2:
            response += f"✅ **Strong Relative Position**\n\n{self.ticker} outperforms sector peers on {wins} out of 3 key metrics. This competitive strength makes it an attractive investment choice within {comparison.get('sector', 'the sector')}.\n\n**Recommendation:** Favorable vs peers - consider as a top pick in sector."
        elif wins == 1:
            response += f"🟡 **Mixed Competitive Position**\n\n{self.ticker} shows competitive performance on some metrics but lags on others. Neither clearly better nor worse than sector average.\n\n**Recommendation:** Adequate but not exceptional vs peers - consider sector alternatives."
        else:
            response += f"🔴 **Weak Relative Position**\n\n{self.ticker} underperforms sector peers across most key metrics. Better investment opportunities likely exist elsewhere in {comparison.get('sector', 'the sector')}.\n\n**Recommendation:** Avoid in favor of stronger sector peers."
        
        return response


class GeneralAnalysisAgent:
    """General financial analysis"""
    
    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm
    
    def analyze(self, query: str) -> str:
        """General analysis"""
        
        with st.status("📊 General Analysis Agent", expanded=True) as status:
            st.write("🔧 Gathering data...")
            metrics_json = get_stock_metrics_tool(self.ticker)
            health_json = calculate_health_score_tool(self.ticker)
            
            st.write("📄 Searching SEC filing...")
            filing_data = search_sec_filing_tool(self.ticker, query)
            
            st.write("🤖 Analyzing...")
            
            try:
                metrics = json.loads(metrics_json)
                health = json.loads(health_json)
            except:
                return "⚠️ Error loading data."
            
            status.update(label="✅ Analysis Complete", state="complete")
        
        # BUILD RESPONSE
        response = f"""## 📊 Financial Overview for {self.ticker}

---

### Company Profile

**Name:** {metrics.get('company_name', self.ticker)}  
**Sector:** {metrics.get('sector', 'Unknown')}  
**Industry:** {metrics.get('industry', 'Unknown')}

---

### Financial Health Score

**Score:** {health.get('score', 0)}/100 ({health.get('rating', 'N/A')})

"""
        
        if health.get('factors'):
            for factor in health['factors']:
                response += f"- {factor}\n"
        
        response += f"""

---

### Key Metrics Summary

| Category | Metric | Value |
|----------|--------|-------|
| **Valuation** | Current Price | ${metrics.get('price', 0)} |
| | P/E Ratio | {metrics.get('pe_ratio', 0)} |
| | Market Cap | ${metrics.get('market_cap_b', 0)}B |
| **Profitability** | Profit Margin | {metrics.get('profit_margin_pct', 0)}% |
| | Operating Margin | {metrics.get('operating_margin_pct', 0)}% |
| **Growth** | Revenue Growth | {metrics.get('revenue_growth_pct', 0)}% |
| | Total Revenue | ${metrics.get('revenue_b', 0)}B |
| **Risk** | Debt/Equity | {metrics.get('debt_to_equity', 0)} |
| | Beta | {metrics.get('beta', 0)} |
| | Current Ratio | {metrics.get('current_ratio', 0)} |

---

### SEC Filing Insights

"""
        
        if filing_data and "No SEC filing" not in filing_data:
            response += filing_data[:800] + "\n\n"
            response += "*[Extracted from uploaded 10-Q]*\n\n"
        else:
            response += "⚠️ Upload 10-Q filing for management discussion and detailed disclosures.\n\n"
        
        response += "---\n\n### Quick Assessment\n\n"
        
        # Overall verdict
        if health.get('score', 0) > 70:
            response += f"✅ **Strong Fundamentals:** {self.ticker} demonstrates solid financial health with a score of {health.get('score', 0)}/100. Metrics suggest a quality company worth considering for investment."
        elif health.get('score', 0) > 50:
            response += f"🟡 **Adequate Fundamentals:** {self.ticker} shows reasonable financial health at {health.get('score', 0)}/100. Suitable for diversified portfolios but not a standout."
        else:
            response += f"🔴 **Weak Fundamentals:** {self.ticker} has concerning financial metrics with a low score of {health.get('score', 0)}/100. Proceed with caution or avoid."
        
        return response
"""
Multi-Agent System using Fireworks Llama 3.1 70B
Each agent specializes in a specific type of financial analysis
REVISED: Clean, professional formatting
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


@st.cache_resource
def get_llm():
    """Initialize Fireworks Llama 3.1 70B"""
    llm = ChatFireworks(
        model="accounts/fireworks/models/llama-v3p3-70b-instruct",
        api_key=st.secrets["FIREWORKS_API_KEY"],
        temperature=0.3,
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
        
        investment_keywords = ['invest', 'buy', 'sell', 'hold', 'price target', 'should i', 'entry', 'exit']
        risk_keywords = ['risk', 'danger', 'concern', 'threat', 'safe', 'risky', 'worry']
        product_keywords = ['product', 'segment', 'business', 'revenue', 'sales', 'performing']
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
    """Investment decisions with clean, structured output"""
    
    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm
    
    def analyze(self, query: str) -> str:
        """Provide investment recommendation"""
        
        with st.status("💼 Investment Advisor Agent", expanded=True) as status:
            st.write("🔧 Gathering financial metrics...")
            metrics_json = get_stock_metrics_tool(self.ticker)
            
            st.write("🎯 Calculating price targets...")
            targets_json = calculate_price_targets_tool(self.ticker, self.portfolio_config['risk_tolerance'])
            
            st.write("🏥 Assessing financial health...")
            health_json = calculate_health_score_tool(self.ticker)
            
            st.write("📄 Checking SEC filing...")
            risks = search_sec_filing_tool(self.ticker, "risks concerns")
            
            st.write("🤖 Generating recommendation...")
            
            # Parse data
            try:
                metrics = json.loads(metrics_json)
                targets = json.loads(targets_json)
                health = json.loads(health_json)
            except:
                return "⚠️ Error parsing financial data. Please try again."
            
            # Get LLM verdict (just the decision + one sentence)
            verdict_prompt = f"""You are a financial advisor. Based on this data for {self.ticker}:

Current Price: ${targets.get('current_price', 0)}
Fair Value: ${targets.get('fair_value', 0)}
Health Score: {health.get('score', 0)}/100
Revenue Growth: {metrics.get('revenue_growth_pct', 0)}%
Profit Margin: {metrics.get('profit_margin_pct', 0)}%

Provide ONLY: "[BUY/HOLD/WAIT] - [one clear sentence why]"

Be concise. Example: "BUY - Strong fundamentals with 55% upside to fair value justify entry at current levels."
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
        
        # Determine recommendation color
        if 'BUY' in verdict.upper():
            verdict_emoji = "✅"
        elif 'WAIT' in verdict.upper():
            verdict_emoji = "⏸️"
        else:
            verdict_emoji = "🔄"
        
        response = f"""## {verdict_emoji} Investment Recommendation for {self.ticker}

**{verdict}**

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
        
        # Entry strategy based on risk tolerance
        if self.portfolio_config['risk_tolerance'] == 'Conservative':
            response += f"""- **Conservative Entry:** ${targets.get('conservative_entry', 0)} (recommended for you)
- **Moderate Entry:** ${targets.get('moderate_entry', 0)}
- **Aggressive Entry:** ${targets.get('aggressive_entry', 0)}
"""
        elif self.portfolio_config['risk_tolerance'] == 'Aggressive':
            response += f"""- **Aggressive Entry:** ${targets.get('aggressive_entry', 0)} (recommended for you)
- **Moderate Entry:** ${targets.get('moderate_entry', 0)}
- **Conservative Entry:** ${targets.get('conservative_entry', 0)}
"""
        else:  # Moderate
            response += f"""- **Moderate Entry:** ${targets.get('moderate_entry', 0)} (recommended for you)
- **Conservative Entry:** ${targets.get('conservative_entry', 0)} (safer)
- **Aggressive Entry:** ${targets.get('aggressive_entry', 0)} (higher risk)
"""
        
        response += f"""- **Target Exit:** ${targets.get('target_exit', 0)} (+{round((targets.get('target_exit', 0) / current_price - 1) * 100, 1) if current_price > 0 else 0}%)
- **Stop Loss:** ${targets.get('stop_loss', 0)} (-8% protection)

---

## 📈 Key Financial Metrics

"""
        
        # Add metrics table
        response += f"""| Metric | {self.ticker} | Assessment |
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
        
        # Build action plan based on analysis
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
3. **Alternative:** If must enter now, use smaller position ({self.portfolio_config['portfolio_allocation'] / 2}%)
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
        
        response += f"""

---

**Investment Profile:** {self.portfolio_config['risk_tolerance']} Risk | {self.portfolio_config['investment_horizon']} Horizon

⚠️ *This is analysis based on current data, not personalized financial advice. Consult a qualified financial advisor before making investment decisions.*
"""
        
        return response


class RiskAnalysisAgent:
    """Risk assessment with structured output"""
    
    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm
    
    def analyze(self, query: str) -> str:
        """Comprehensive risk analysis"""
        
        with st.status("🛡️ Risk Analysis Agent", expanded=True) as status:
            st.write("🔧 Gathering financial metrics...")
            metrics_json = get_stock_metrics_tool(self.ticker)
            
            st.write("📄 Extracting SEC filing risks...")
            filing_risks = search_sec_filing_tool(self.ticker, "risk factors threats concerns regulatory")
            
            st.write("🤖 Analyzing risk profile...")
            
            # Parse metrics
            try:
                metrics = json.loads(metrics_json)
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
        if risks and "No SEC filing" not in filing_risks:
            response += filing_risks[:800] + "...\n\n"
        else:
            response += "*No SEC filing uploaded. Upload 10-Q for detailed risk disclosures.*\n\n"
        
        response += "---\n\n### Overall Risk Assessment\n\n"
        
        # Calculate overall risk
        risk_score = 0
        if dte > 150: risk_score += 2
        elif dte > 80: risk_score += 1
        
        if current_ratio < 1.0: risk_score += 2
        elif current_ratio < 1.5: risk_score += 1
        
        if beta > 1.5: risk_score += 2
        elif beta > 1.0: risk_score += 1
        
        if risk_score >= 4:
            rating = "🔴 **HIGH RISK**"
            explanation = "Multiple concerning financial indicators suggest elevated investment risk."
        elif risk_score >= 2:
            rating = "🟡 **MODERATE RISK**"
            explanation = "Some risk factors present. Suitable for moderate risk tolerance investors."
        else:
            rating = "🟢 **LOW RISK**"
            explanation = "Strong financial metrics suggest relatively low investment risk."
        
        response += f"{rating}\n\n{explanation}\n\n"
        
        response += f"""---

### Risk Mitigation Strategies

1. **Position Sizing:** Limit to {self.portfolio_config['portfolio_allocation']}% of portfolio to manage exposure
2. **Stop Loss:** Set at ${targets.get('stop_loss', 0)} to limit downside
3. **Diversification:** Balance with lower-beta stocks in other sectors
4. **Monitoring:** Review quarterly earnings and filing updates

---

*Risk profile based on {self.portfolio_config['risk_tolerance']} tolerance setting.*
"""
        
        return response


class ProductAnalysisAgent:
    """Product/segment analysis"""
    
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
            products = search_sec_filing_tool(self.ticker, "product segment revenue sales")
            
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
            response += "🟢 Strong growth indicates successful product portfolio\n\n"
        elif growth > 5:
            response += "🟡 Moderate growth suggests stable but mature products\n\n"
        else:
            response += "🔴 Weak growth may indicate product challenges\n\n"
        
        response += f"""**Profit Margin:** {margin}% """
        if margin > 20:
            response += "🟢 Premium margins suggest strong product differentiation\n\n"
        elif margin > 10:
            response += "🟡 Healthy margins but facing some competitive pressure\n\n"
        else:
            response += "🔴 Thin margins indicate commoditization risk\n\n"
        
        response += "---\n\n### Product/Segment Details from SEC Filing\n\n"
        
        # Add filing data
        if products and "No SEC filing" not in products:
            response += products[:1000] + "\n\n"
        else:
            response += f"""⚠️ **No SEC filing uploaded for {self.ticker}**

To get detailed product segment breakdowns:
1. Upload the latest 10-Q filing in the sidebar
2. The MD&A section contains product category revenue data
3. I'll extract specific product performance metrics

For now, the overall metrics suggest:
- {"Strong" if growth > 10 and margin > 15 else "Moderate" if growth > 0 else "Weak"} product portfolio performance
- {"Multiple growth drivers likely" if growth > 15 else "Stable product mix" if growth > 5 else "Product challenges possible"}

"""
        
        response += f"""---

### Investment Implications

Based on {growth}% revenue growth and {margin}% margins, {self.ticker}'s product portfolio appears {"strong and competitive" if growth > 10 and margin > 15 else "stable but mature" if growth > 0 else "challenged"}.
"""
        
        return response


class PeerComparisonAgent:
    """Peer comparison with tables"""
    
    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm
    
    def analyze(self, query: str) -> str:
        """Compare to peers"""
        
        with st.status("🏆 Peer Comparison Agent", expanded=True) as status:
            st.write("🔧 Gathering data...")
            metrics_json = get_stock_metrics_tool(self.ticker)
            comparison_json = get_peer_comparison_tool(self.ticker)
            
            st.write("🤖 Analyzing competitive position...")
            
            try:
                metrics = json.loads(metrics_json)
                comparison = json.loads(comparison_json)
            except:
                return "⚠️ Error loading comparison data."
            
            status.update(label="✅ Peer Comparison Complete", state="complete")
        
        # BUILD RESPONSE
        ticker_pe = comparison.get('ticker_pe', 0)
        sector_pe = comparison.get('sector_avg_pe', 0)
        ticker_margin = comparison.get('ticker_margin_pct', 0)
        sector_margin = comparison.get('sector_avg_margin_pct', 0)
        ticker_growth = comparison.get('ticker_growth_pct', 0)
        sector_growth = comparison.get('sector_avg_growth_pct', 0)
        
        response = f"""## 🏆 Competitive Analysis for {self.ticker}

**Sector:** {comparison.get('sector', 'Unknown')}

---

### Valuation Comparison

"""
        
        if ticker_pe > 0 and sector_pe > 0:
            pe_diff = round(((ticker_pe - sector_pe) / sector_pe * 100), 1)
            if pe_diff > 15:
                response += f"🔴 **Premium Valuation:** {self.ticker}'s P/E of {ticker_pe} is {abs(pe_diff)}% above sector average of {sector_pe}\n\n"
            elif pe_diff > -15:
                response += f"🟡 **Market Valuation:** {self.ticker}'s P/E of {ticker_pe} is roughly in line with sector average of {sector_pe}\n\n"
            else:
                response += f"🟢 **Discount Valuation:** {self.ticker}'s P/E of {ticker_pe} is {abs(pe_diff)}% below sector average of {sector_pe}\n\n"
        
        response += "### Profitability Comparison\n\n"
        
        if ticker_margin > sector_margin:
            margin_diff = round(ticker_margin - sector_margin, 1)
            response += f"🟢 **Above Average:** {self.ticker}'s {ticker_margin}% margin exceeds sector average of {sector_margin}% by {margin_diff} percentage points\n\n"
        else:
            margin_diff = round(sector_margin - ticker_margin, 1)
            response += f"🔴 **Below Average:** {self.ticker}'s {ticker_margin}% margin trails sector average of {sector_margin}% by {margin_diff} percentage points\n\n"
        
        response += "### Growth Comparison\n\n"
        
        if ticker_growth > sector_growth:
            growth_diff = round(ticker_growth - sector_growth, 1)
            response += f"🟢 **Faster Growth:** {self.ticker}'s {ticker_growth}% growth outpaces sector average of {sector_growth}%\n\n"
        else:
            growth_diff = round(sector_growth - ticker_growth, 1)
            response += f"🔴 **Slower Growth:** {self.ticker}'s {ticker_growth}% growth lags sector average of {sector_growth}%\n\n"
        
        response += "---\n\n### Peer Data\n\n"
        
        # Peer table
        if comparison.get('peers'):
            response += "| Ticker | P/E | Profit Margin | Revenue Growth |\n"
            response += "|--------|-----|---------------|----------------|\n"
            for peer in comparison['peers'][:4]:
                response += f"| {peer['ticker']} | {peer.get('pe', 'N/A')} | {peer.get('profit_margin_pct', 'N/A')}% | {peer.get('revenue_growth_pct', 'N/A')}% |\n"
            response += f"| **{self.ticker}** | **{ticker_pe}** | **{ticker_margin}%** | **{ticker_growth}%** |\n"
            response += f"| *Sector Avg* | *{sector_pe}* | *{sector_margin}%* | *{sector_growth}%* |\n\n"
        
        response += "---\n\n### Investment Verdict\n\n"
        
        # Determine if better than peers
        beats_valuation = ticker_pe < sector_pe if ticker_pe > 0 and sector_pe > 0 else False
        beats_profitability = ticker_margin > sector_margin
        beats_growth = ticker_growth > sector_growth
        
        wins = sum([beats_valuation, beats_profitability, beats_growth])
        
        if wins >= 2:
            response += f"✅ **Strong Relative Position:** {self.ticker} outperforms sector peers on {wins} out of 3 key metrics, making it an attractive choice within {comparison.get('sector', 'the sector')}."
        elif wins == 1:
            response += f"🟡 **Mixed Position:** {self.ticker} shows competitive performance on some metrics but lags on others. Consider sector alternatives."
        else:
            response += f"🔴 **Weak Position:** {self.ticker} underperforms sector peers on most metrics. Better opportunities may exist elsewhere in {comparison.get('sector', 'the sector')}."
        
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

---

### Quick Assessment

"""
        
        # Quick verdict
        if health.get('score', 0) > 70:
            response += f"✅ **Strong Fundamentals:** {self.ticker} demonstrates solid financial health with a score of {health.get('score', 0)}/100."
        elif health.get('score', 0) > 50:
            response += f"🟡 **Adequate Fundamentals:** {self.ticker} shows reasonable financial health at {health.get('score', 0)}/100."
        else:
            response += f"🔴 **Weak Fundamentals:** {self.ticker} has concerning financial metrics with a low score of {health.get('score', 0)}/100."
        
        return response
"""
Multi-Agent System using Fireworks Llama 3.1 70B
Agents + orchestration for FinChat AI
"""

import streamlit as st
from langchain_fireworks import ChatFireworks
from typing import Dict
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
# SUPPORT / SHARED HELPERS
# ============================================================================

SUPPORTED_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA',
    'AMZN', 'JPM', 'V', 'JNJ', 'WMT', 'PG', 'NFLX', 'DIS'
]


def supported_companies_markdown() -> str:
    """Markdown string listing supported companies (for sidebar + errors)."""
    return """
**Technology**  
- AAPL – Apple Inc.  
- MSFT – Microsoft Corporation  
- GOOGL – Alphabet Inc.  
- META – Meta Platforms  
- NVDA – NVIDIA Corporation  
- TSLA – Tesla Inc.  
- AMZN – Amazon.com Inc.  

**Consumer & Media**  
- WMT – Walmart  
- PG – Procter & Gamble  
- NFLX – Netflix  
- DIS – The Walt Disney Company  

**Financial Services**  
- JPM – JPMorgan Chase & Co.  
- V – Visa Inc.  

**Healthcare**  
- JNJ – Johnson & Johnson  
"""


# ============================================================================
# TICKER EXTRACTION HELPERS
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

    available_tickers = SUPPORTED_TICKERS

    specific_keywords = [
        'STOCK', 'PRICE', 'COMPANY', 'INVEST', 'BUY', 'SELL',
        'ANALYSIS', 'ABOUT', 'TELL ME', 'SHOULD I'
    ]

    query_upper = query.upper()
    is_specific_query = any(keyword in query_upper for keyword in specific_keywords)

    # 1) Ticker symbols
    for ticker in available_tickers:
        pattern = r'\b' + re.escape(ticker) + r'\b'
        if re.search(pattern, query_upper):
            return ticker, True

    # 2) Company names
    for company_name in sorted(ticker_map.keys(), key=len, reverse=True):
        if company_name in query_upper:
            ticker = ticker_map[company_name]
            if ticker in available_tickers:
                return ticker, True

    # 3) Unsupported but common companies
    unsupported_companies = [
        'BOEING', 'BA', 'CITI', 'CITIBANK', 'CITIGROUP', 'BANK OF AMERICA',
        'BOA', 'BAC', 'WELLS FARGO', 'WFC', 'GOLDMAN', 'GOLDMAN SACHS', 'GS',
        'MORGAN STANLEY', 'MS', 'AMD', 'INTEL', 'INTC', 'IBM', 'ORACLE', 'ORCL',
        'UBER', 'LYFT', 'AIRBNB', 'ABNB', 'SNAP', 'TWITTER', 'FORD', 'GM',
        'GENERAL MOTORS', 'COCA COLA', 'COKE', 'KO', 'PEPSI', 'PEP',
        'MCDONALD', 'MCD', 'STARBUCKS', 'SBUX', 'NIKE', 'NKE'
    ]

    if is_specific_query:
        for unsupported in unsupported_companies:
            if unsupported in query_upper:
                return None, True  # Found company but not supported

    # 4) Fallback to current dropdown ticker
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

    # Company names → tickers
    for company_name in sorted(ticker_map.keys(), key=len, reverse=True):
        if company_name in query_upper and ticker_map[company_name] not in found_tickers:
            found_tickers.append(ticker_map[company_name])
            if len(found_tickers) >= 2:
                break

    # Explicit ticker symbols
    if len(found_tickers) < 2:
        all_tickers = SUPPORTED_TICKERS
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
        """Classify user query into one of: investment, risk, product, comparison, general."""
        query_lower = query.lower()

        # Comparison-oriented language
        comparison_keywords = [
            'compare', 'comparison', 'vs', 'versus',
            'better than', 'relative to', 'stand relative',
            'stack up', 'peer', 'peers', 'competitor', 'competitors'
        ]

        # Make "invest" safer so it doesn't hit "investor"
        investment_keywords = [
            'should i invest', 'invest in', 'investing in',
            'buy', 'sell', 'hold', 'price target',
            'good buy', 'what should i do',
            'have stocks', 'stock price', 'price of'
        ]

        risk_keywords = [
            'risk', 'risks', 'danger', 'concern', 'threat',
            'safe', 'safety', 'risky', 'worry', 'worried'
        ]

        product_keywords = [
            'product', 'products', 'segment', 'segments',
            'business mix', 'business segments',
            'revenue by segment', 'sales mix',
            'advertising', 'ad revenue'
        ]

        def contains_any(text: str, keywords) -> bool:
            return any(kw in text for kw in keywords)

        # Priority: comparison → investment → risk → product → general
        if contains_any(query_lower, comparison_keywords):
            return 'comparison'
        elif contains_any(query_lower, investment_keywords):
            return 'investment'
        elif contains_any(query_lower, risk_keywords):
            return 'risk'
        elif contains_any(query_lower, product_keywords):
            return 'product'
        else:
            return 'general'

    def route_query(self, query: str) -> str:
        """Route to appropriate specialized agent"""
        intent = self.classify_intent(query)

        if intent == 'investment':
            return InvestmentAdvisorAgent(
                self.ticker, self.portfolio_config, self.llm
            ).analyze(query)
        elif intent == 'risk':
            return RiskAnalysisAgent(
                self.ticker, self.portfolio_config, self.llm
            ).analyze(query)
        elif intent == 'product':
            return ProductAnalysisAgent(
                self.ticker, self.portfolio_config, self.llm
            ).analyze(query)
        elif intent == 'comparison':
            return PeerComparisonAgent(
                self.ticker, self.portfolio_config, self.llm
            ).analyze(query)
        else:
            return GeneralAnalysisAgent(
                self.ticker, self.portfolio_config, self.llm
            ).analyze(query)


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
            return f"""## ⚠️ Company Not Supported

Sorry, I don't have data for the company you mentioned.

FinChat AI currently supports these companies:

{supported_companies_markdown()}

💡 **Tip:** Select one of these companies from the dropdown or ask about them by name!
"""

        with st.status("💼 Investment Advisor Agent", expanded=True) as status:
            st.write(f"🎯 Analyzing {analyzed_ticker}...")
            st.write("🔧 Gathering financial metrics...")
            metrics_json = get_stock_metrics_tool(analyzed_ticker)

            st.write("🎯 Calculating price targets...")
            targets_json = calculate_price_targets_tool(
                analyzed_ticker,
                self.portfolio_config['risk_tolerance']
            )

            st.write("🏥 Assessing financial health...")
            health_json = calculate_health_score_tool(analyzed_ticker)

            st.write("📄 Checking SEC filing...")
            _ = search_sec_filing_tool(analyzed_ticker, "risks concerns")  # not used directly here

            st.write("🤖 Generating recommendation...")

            # Parse data
            try:
                metrics = json.loads(metrics_json)
                targets = json.loads(targets_json)
                health = json.loads(health_json)
            except Exception:
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
            owns_stock = any(
                phrase in query_lower for phrase in
                ['i have', 'i own', 'holding', 'my stocks']
            )

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

        current_price = targets.get('current_price', 0)
        fair_value = targets.get('fair_value', 0)
        upside = targets.get('upside_percent', 0)
        health_score = health.get('score', 0)
        health_rating = health.get('rating', 'N/A')

        verdict_upper = verdict.upper()
        if 'BUY' in verdict_upper or 'HOLD' in verdict_upper:
            verdict_emoji = "✅"
        elif 'WAIT' in verdict_upper:
            verdict_emoji = "⏸️"
        else:
            verdict_emoji = "🔄"

        if owns_stock:
            intro = f"**You own {analyzed_ticker}:**\n\n"
        elif 'conservative' in query_lower:
            intro = "**For Conservative Investors:**\n\n"
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
# RISK ANALYSIS AGENT
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
            return f"""## ⚠️ Company Not Supported

Sorry, I don't have risk analysis data for the company you mentioned.

Supported companies:

{supported_companies_markdown()}

Select a supported company from the dropdown or ask about one by name.
"""

        query_lower = query.lower()
        quote_mode = ("quote" in query_lower) and ("sentence" in query_lower)

        with st.status("🛡️ Risk Analysis Agent", expanded=True) as status:
            st.write(f"🎯 Analyzing {analyzed_ticker}...")
            st.write("🔧 Gathering financial metrics...")
            metrics_json = get_stock_metrics_tool(analyzed_ticker)

            st.write("🎯 Calculating risk thresholds...")
            targets_json = calculate_price_targets_tool(
                analyzed_ticker,
                self.portfolio_config['risk_tolerance']
            )

            st.write("🏥 Computing health score...")
            health_json = calculate_health_score_tool(analyzed_ticker)

            st.write("📄 Extracting SEC filing risks...")
            # same retrieval, but prompt will depend on quote_mode
            filing_risks = search_sec_filing_tool(
                analyzed_ticker,
                "risk factors threats concerns regulatory competition macroeconomic"
            )

            st.write("🤖 Analyzing risk profile...")

            try:
                metrics = json.loads(metrics_json)
                targets = json.loads(targets_json)
                health = json.loads(health_json)
            except Exception:
                return "⚠️ Error loading data."

            if 'error' in metrics:
                return "⚠️ Unable to fetch risk data for this company."

            status.update(label="✅ Risk Analysis Complete", state="complete")

        # --- Quantitative interpretation ---

        beta = metrics.get('beta', 0)
        dte = metrics.get('debt_to_equity', 0)
        growth = metrics.get('revenue_growth_pct', 0)
        margin = metrics.get('profit_margin_pct', 0)
        health_score = health.get('score', 0)
        health_rating = health.get('rating', 'N/A')

        # Simple risk level heuristic
        risk_score = 0
        if beta > 1.3:
            risk_score += 2
        elif beta > 1.0:
            risk_score += 1

        if dte > 150:
            risk_score += 2
        elif dte > 80:
            risk_score += 1

        if growth < 0:
            risk_score += 2
        elif growth < 5:
            risk_score += 1

        if health_score < 40:
            risk_score += 2
        elif health_score < 60:
            risk_score += 1

        if risk_score <= 2:
            overall_risk = "Low–Moderate"
            risk_emoji = "🟢"
        elif risk_score <= 4:
            overall_risk = "Moderate"
            risk_emoji = "🟡"
        else:
            overall_risk = "Elevated"
            risk_emoji = "🔴"

        # --- Choose LLM prompt depending on quote_mode ---

        if quote_mode:
            # SPECIAL MODE: user explicitly asked to quote 1–2 sentences
            risk_prompt = f"""You are an equity analyst.

The user asked:
\"\"\"{query}\"\"\"


Here are excerpts from the company's SEC filing (Risk Factors / MD&A / related sections):

\"\"\"{filing_risks[:8000]}\"\"\"


Your task:

1. Find **1–2 sentences from the filing text that directly mention competition or competitive risks.**
2. Copy those sentences **exactly** as they appear in the filing (no rewriting, no summarizing).
3. Then explain those sentences in **very simple language** for someone with no finance background.

Return **markdown** in exactly this structure:

### Quoted sentences
- "First sentence from the filing."
- "Second sentence from the filing."  (omit if you only find one)

### Simple explanation
- A short paragraph (3–5 sentences) explaining what these risks mean in everyday terms.

Do NOT add any other sections or commentary.
"""
        else:
            # DEFAULT MODE: full narrative risk analysis using filing + numbers
            risk_prompt = f"""You are an equity risk analyst.

User question:
\"\"\"{query}\"\"\"


Numeric snapshot for {analyzed_ticker}:
- Beta (volatility vs market): {beta}
- Debt-to-Equity: {dte}
- Revenue growth (YoY): {growth}%
- Profit margin: {margin}%
- Health score: {health_score}/100 ({health_rating})

Here are excerpts from the company's SEC filing (Risk Factors / MD&A / related sections):

\"\"\"{filing_risks[:8000]}\"\"\"  # may already be formatted markdown

Based on ALL of this, write a concise markdown answer with:

1. **Overall Risk Verdict** – 1 short sentence starting with a label like "Low", "Moderate", or "High".
2. **Key Risk Buckets** – 3–6 bullets tagged as things like *Competitive*, *Regulatory*, *Macro*, *Execution*, *Financial*, etc.
3. **SEC Filing Highlights** – 2–3 bullets that paraphrase, and where appropriate quote a short phrase from the filing in *italics*.
4. **What This Means for a {self.portfolio_config['risk_tolerance']} Investor** – 2–3 sentences of practical guidance (do NOT give explicit buy/sell advice, just describe the risk stance).

Focus on clarity. Avoid repeating raw numbers; interpret them.
"""

        llm_resp = self.llm.invoke(risk_prompt)
        narrative = llm_resp.content.strip()

        # If we’re in quote mode, just return the quoted block (no extra wrapper)
        if quote_mode:
            return f"## 🟢 Competition Risk from SEC Filing for {analyzed_ticker}\n\n" + narrative

        # --- Build full markdown (default mode) ---

        risk_table = f"""## {risk_emoji} Risk Assessment for {analyzed_ticker}

### 🔢 Quantitative Risk Snapshot

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Beta** | {beta} | {'🟢 Below market volatility' if beta < 1.0 else '🟡 Around market volatility' if beta < 1.3 else '🔴 Higher than market, more volatile'} |
| **Debt / Equity** | {dte} | {'🟢 Conservative leverage' if dte < 50 else '🟡 Moderate leverage' if dte < 150 else '🔴 High leverage'} |
| **Revenue Growth (YoY)** | {growth}% | {'🟢 Healthy growth' if growth > 10 else '🟡 Modest growth' if growth > 0 else '🔴 Flat/declining'} |
| **Profit Margin** | {margin}% | {'🟢 Strong profitability' if margin > 20 else '🟡 Reasonable margins' if margin > 10 else '🔴 Thin margins'} |
| **Health Score** | {health_score}/100 | {health_rating} |

**Overall Risk Level (blended):** **{overall_risk}**
"""

        response = risk_table + "\n\n---\n\n" + narrative
        return response


# ============================================================================
# PRODUCT ANALYSIS AGENT
# ============================================================================

class ProductAnalysisAgent:
    """Product & segment analysis with ticker extraction"""

    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm

    def analyze(self, query: str) -> str:
        analyzed_ticker, found_in_query = extract_ticker_from_query(query, self.ticker)

        # Unsupported company handling
        if analyzed_ticker is None and found_in_query:
            return f"""## ⚠️ Company Not Supported

Sorry, I don't have product/segment analysis data for the company you mentioned.

Supported companies:

{supported_companies_markdown()}

Select a supported company from the dropdown or ask about one by name.
"""

        with st.status("📦 Product Analysis Agent", expanded=True) as status:
            status.write(f"🎯 Analyzing product mix for {analyzed_ticker}...")
            status.write("🔧 Gathering financial metrics...")
            metrics_json = get_stock_metrics_tool(analyzed_ticker)

            status.write("🏥 Checking overall financial health...")
            health_json = calculate_health_score_tool(analyzed_ticker)

            status.write("📄 Pulling product & segment discussion from SEC filing...")
            # Focus the RAG query on business / products / segments
            sec_product_text = search_sec_filing_tool(
                analyzed_ticker,
                f"business overview products services segments revenue drivers competition user engagement advertising subscriptions {query}"
            )

            try:
                metrics = json.loads(metrics_json)
                health = json.loads(health_json)
            except Exception:
                return "⚠️ Error loading product data."

            if "error" in metrics:
                return "⚠️ Unable to fetch product metrics for this company."

            status.update(label="✅ Product Analysis Complete", state="complete")

        # --- Pull out some useful numbers/labels ---

        company_name = metrics.get("company_name", analyzed_ticker)
        sector = metrics.get("sector", "Unknown sector")
        industry = metrics.get("industry", "Unknown industry")
        revenue_growth = metrics.get("revenue_growth_pct", 0)
        profit_margin = metrics.get("profit_margin_pct", 0)
        operating_margin = metrics.get("operating_margin_pct", 0)
        health_score = health.get("score", 0)
        health_rating = health.get("rating", "N/A")

        # --- Build LLM prompt focused on products & segments ---

        product_prompt = f"""You are an equity research analyst focused on **product and segment analysis**.

User question:
\"\"\"{query}\"\"\"

Company: {company_name} ({analyzed_ticker})
Sector / Industry: {sector} / {industry}

Key financial context:
- Revenue growth (YoY): {revenue_growth}%
- Profit margin: {profit_margin}%
- Operating margin: {operating_margin}%
- Health score: {health_score}/100 ({health_rating})

Here are excerpts from the company's most recent 10-Q / 10-K filing related to business, products, and segments:

\"\"\"{sec_product_text[:8000]}\"\"\"


Using ONLY this information plus your general understanding of the company (but **do not invent precise numeric breakdowns**), write a concise markdown answer with these sections:

1. **Product & Segment Snapshot**  
   - 3–6 bullets listing the main products or business segments and what they do (e.g., for META, "Family of Apps", "Reality Labs").

2. **What’s Driving Growth**  
   - 2–5 bullets describing which products/segments are currently driving revenue growth or engagement and why.  
   - If the filing mentions ads, subscriptions, cloud, hardware, or specific apps, refer to them explicitly.

3. **Areas of Weakness or Heavy Investment**  
   - 2–5 bullets on segments that are under pressure, loss-making, cyclical, or where the company is heavily investing for future upside (e.g., VR/AR, metaverse, new chips, etc.).

4. **Takeaway for a {self.portfolio_config['risk_tolerance']} Investor**  
   - A short paragraph (3–5 sentences) summarizing how diversified the business is across products/segments and how the product mix affects risk and upside.  
   - Do **not** give explicit buy/sell advice; just explain risk/return profile.

Keep the tone clear and concrete, and avoid buzzwords or generic filler.
"""

        llm_resp = self.llm.invoke(product_prompt)
        narrative = llm_resp.content.strip()

        # Simple header + quick snapshot table
        header = f"""## 📦 Product & Segment Analysis for {analyzed_ticker}

**Company:** {company_name}  
**Sector / Industry:** {sector} / {industry}  

### 🔢 High-Level Business Snapshot

| Metric | Value |
|--------|-------|
| Revenue Growth (YoY) | {revenue_growth}% |
| Profit Margin | {profit_margin}% |
| Operating Margin | {operating_margin}% |
| Health Score | {health_score}/100 ({health_rating}) |

---

"""

        return header + narrative

# ============================================================================
# PEER COMPARISON AGENT
# ============================================================================

class PeerComparisonAgent:
    """Peer comparison with head-to-head & sector support"""

    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm

    def analyze(self, query: str) -> str:
        # Try to pull one or two tickers out of the question
        ticker1, ticker2 = extract_tickers_from_query(query, self.ticker)

        # If the user mentioned an unsupported company explicitly
        if ticker1 is None:
            return f"""## ⚠️ Company Not Supported

I couldn't find a supported ticker in your question.

Supported companies:

{supported_companies_markdown()}

Try something like:
- "Compare AAPL vs MSFT"
- "Is META better than GOOGL for growth?"
"""

        # If two tickers found → true head-to-head
        if ticker2:
            return self.head_to_head_comparison(ticker1, ticker2, query)
        # Otherwise compare ticker1 vs its sector peers
        else:
            return self.sector_comparison(ticker1, query)

    # ------------------------------------------------------------------ #
    # HEAD-TO-HEAD
    # ------------------------------------------------------------------ #

    def head_to_head_comparison(self, ticker1: str, ticker2: str, query: str) -> str:
        with st.status("🏆 Peer Comparison Agent", expanded=True) as status:
            status.write(f"🔍 Comparing {ticker1} vs {ticker2}...")

            status.write("📊 Fetching core metrics...")
            m1_json = get_stock_metrics_tool(ticker1)
            m2_json = get_stock_metrics_tool(ticker2)

            status.write("🏥 Fetching health scores...")
            h1_json = calculate_health_score_tool(ticker1)
            h2_json = calculate_health_score_tool(ticker2)

            try:
                m1 = json.loads(m1_json)
                m2 = json.loads(m2_json)
                h1 = json.loads(h1_json)
                h2 = json.loads(h2_json)
            except Exception:
                return "⚠️ Error loading comparison data."

            if "error" in m1 or "error" in m2:
                return f"""## ⚠️ Peer Data Not Available

I wasn't able to fetch comparison data for this pair.

Try comparing among these tickers (built-in peer set):

- AAPL, MSFT, GOOGL, META, NVDA, TSLA, AMZN, NFLX, DIS, V, JPM
"""

            status.write("🤖 Generating comparison narrative...")

        # Quick helpers
        def safe(x, key, default=0.0):
            return x.get(key, default) or default

        # Build a compact numeric snapshot
        table = f"""## 🏆 Head-to-Head: {ticker1} vs {ticker2}

### 🔢 Core Metrics Snapshot

| Metric | {ticker1} | {ticker2} |
|--------|-----------|-----------|
| P/E Ratio | {safe(m1, 'pe_ratio'):.1f} | {safe(m2, 'pe_ratio'):.1f} |
| Forward P/E | {safe(m1, 'forward_pe'):.1f} | {safe(m2, 'forward_pe'):.1f} |
| Revenue Growth (YoY) | {safe(m1, 'revenue_growth_pct'):.1f}% | {safe(m2, 'revenue_growth_pct'):.1f}% |
| Profit Margin | {safe(m1, 'profit_margin_pct'):.1f}% | {safe(m2, 'profit_margin_pct'):.1f}% |
| Operating Margin | {safe(m1, 'operating_margin_pct'):.1f}% | {safe(m2, 'operating_margin_pct'):.1f}% |
| Debt / Equity | {safe(m1, 'debt_to_equity'):.1f} | {safe(m2, 'debt_to_equity'):.1f} |
| Health Score | {safe(h1, 'score', 0):.0f}/100 | {safe(h2, 'score', 0):.0f}/100 |
"""

        # LLM prompt for interpretation
        prompt = f"""You are an equity analyst comparing **{ticker1}** vs **{ticker2}**.

User question:
\"\"\"{query}\"\"\"


Here are summary metrics:

{table}

Write a concise markdown answer with:

1. **Overview** – 2–3 sentences summarizing which company looks stronger overall and why.
2. **Strengths of {ticker1}** – 3–5 short bullets.
3. **Strengths of {ticker2}** – 3–5 short bullets.
4. **Risk Profile Comparison** – 2–4 bullets comparing volatility, leverage, and business concentration.
5. **Fit for a {self.portfolio_config['risk_tolerance']} Investor** – 3–5 sentences describing which type of investor might prefer each stock (but do NOT give explicit buy/sell instructions).

Be specific but plain-English. Do not invent extra numbers beyond what you see above.
"""

        resp = self.llm.invoke(prompt)
        narrative = resp.content.strip()

        return table + "\n\n" + narrative

    # ------------------------------------------------------------------ #
    # SINGLE-TICKER VS SECTOR PEERS
    # ------------------------------------------------------------------ #

    def sector_comparison(self, ticker: str, query: str) -> str:
        with st.status("🏆 Peer Comparison Agent", expanded=True) as status:
            status.write(f"🔍 Comparing {ticker} vs its sector peers...")

            status.write("📊 Fetching peer data...")
            peer_json = get_peer_comparison_tool(ticker)
            m_json = get_stock_metrics_tool(ticker)

            try:
                peer = json.loads(peer_json)
                metrics = json.loads(m_json)
            except Exception:
                return "⚠️ Error loading sector comparison data."

            if "error" in peer:
                # Fallback: no structured peer data
                return f"""## 🏆 Peer & Sector View for {ticker}

Structured peer comparison data isn't available for this ticker yet.

You can still compare it manually by asking:
- "Compare {ticker} vs MSFT"
- "How does {ticker} compare to GOOGL on growth and margins?"
"""

            status.write("🤖 Analyzing peer set...")

        # Build peer table
        rows = []
        for p in peer.get("peers", []):
            rows.append(
                f"| {p['ticker']} | {p['pe']:.1f} | "
                f"{p['profit_margin_pct']:.1f}% | {p['revenue_growth_pct']:.1f}% |"
            )

        peer_table = "\n".join(rows) if rows else "_Peer details not available_"

        header = f"""## 🏆 {ticker} vs Sector Peers

**Sector:** {peer.get('sector', 'Unknown')}  

### 🔢 Snapshot vs Sector

| Metric | {ticker} | Sector Avg |
|--------|----------|-----------|
| P/E Ratio | {peer.get('ticker_pe', 0):.1f} | {peer.get('sector_avg_pe', 0):.1f} |
| Profit Margin | {peer.get('ticker_margin_pct', 0):.1f}% | {peer.get('sector_avg_margin_pct', 0):.1f}% |
| Revenue Growth (YoY) | {peer.get('ticker_growth_pct', 0):.1f}% | {peer.get('sector_avg_growth_pct', 0):.1f}% |

### 📊 Key Peers

| Ticker | P/E | Profit Margin | Revenue Growth |
|--------|-----|---------------|----------------|
{peer_table}

---
"""

        prompt = f"""You are an equity analyst.

User question:
\"\"\"{query}\"\"\"


Here is a summary of how **{ticker}** compares to its sector peers:

{header}

Write a concise markdown explanation with:

1. **Positioning vs Peers** – 3–5 bullets on valuation (P/E), growth, and profitability vs sector averages.
2. **Competitive Edge / Weakness** – 3–5 bullets on where {ticker} stands out or lags (e.g., higher margins, faster growth, expensive valuation, etc.).
3. **Takeaway for a {self.portfolio_config['risk_tolerance']} Investor** – 3–5 sentences about what this peer positioning means in terms of risk and potential reward.

Do not reprint the tables; interpret them in plain English.
"""

        resp = self.llm.invoke(prompt)
        narrative = resp.content.strip()

        return header + "\n" + narrative


# ============================================================================
# GENERAL ANALYSIS AGENT
# ============================================================================

class GeneralAnalysisAgent:
    """General analysis with ticker extraction"""

    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm

    def analyze(self, query: str) -> str:
        analyzed_ticker, found_in_query = extract_ticker_from_query(query, self.ticker)

        if analyzed_ticker is None and found_in_query:
            return f"""## ⚠️ Company Not Supported

Sorry, I don't have data for the company you mentioned.

FinChat AI currently supports these companies:

{supported_companies_markdown()}

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

            except Exception:
                return "⚠️ Error loading data."

            status.update(label="✅ Analysis Complete", state="complete")

        response = f"""## 📊 Financial Overview for {analyzed_ticker}

**Name:** {metrics.get('company_name', analyzed_ticker)}  
**Sector:** {metrics.get('sector', 'Unknown')}

**Health Score:** {health.get('score', 0)}/100 ({health.get('rating', 'N/A')})

### Key Metrics

- Price: ${metrics.get('price', 0)}
- P/E: {metrics.get('pe_ratio', 0)}
- Revenue: ${metrics.get('revenue_b', 0)}B
- Revenue Growth (YoY): {metrics.get('revenue_growth_pct', 0)}%
- Profit Margin: {metrics.get('profit_margin_pct', 0)}%
- Debt/Equity: {metrics.get('debt_to_equity', 0)}
- Beta: {metrics.get('beta', 0)}

---

If you'd like, ask things like:
- "What are the main risks for this company?"
- "How does it compare to its competitors?"
- "Is this suitable for a {self.portfolio_config['risk_tolerance'].lower()} investor?"
"""

        return response

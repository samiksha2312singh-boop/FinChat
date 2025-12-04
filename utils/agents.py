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
        """Classify user query"""
        query_lower = query.lower()

        investment_keywords = [
            'invest', 'buy', 'sell', 'hold', 'price target', 'should i',
            'entry', 'exit', 'good buy', 'what should i do', 'have stocks',
            'stock price', 'price of'
        ]
        risk_keywords = [
            'risk', 'danger', 'concern', 'threat', 'safe', 'risky',
            'worry', 'worried'
        ]
        product_keywords = [
            'product', 'segment', 'business', 'revenue', 'sales',
            'performing', 'advertising'
        ]
        comparison_keywords = [
            'compare', 'competitor', 'peer', 'vs', 'versus',
            'better than', 'or'
        ]

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

cclass RiskAnalysisAgent:
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
    """Product / segment analysis with ticker extraction"""

    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm

    def analyze(self, query: str) -> str:
        analyzed_ticker, found_in_query = extract_ticker_from_query(query, self.ticker)

        if analyzed_ticker is None and found_in_query:
            return "## ⚠️ Company Not Supported\n\nSorry, product analysis is not available for this company."

        with st.status("📦 Product Analysis Agent", expanded=True) as status:
            st.write(f"🎯 Analyzing {analyzed_ticker} products/segments...")
            metrics_json = get_stock_metrics_tool(analyzed_ticker)
            metrics = {}
            try:
                metrics = json.loads(metrics_json)
            except Exception:
                pass

            st.write("📄 Extracting product/segment info from SEC filing...")
            filing_text = search_sec_filing_tool(
                analyzed_ticker,
                "products services segments revenue growth business lines customers platforms advertising"
            )

            status.update(label="✅ Product Analysis Complete", state="complete")

        prompt = f"""You are a fundamental equity analyst.

The user asked:
\"\"\"{query}\"\"\"


Here is a numeric snapshot (if present):
{json.dumps(metrics, indent=2)}


Here are excerpts from the company's SEC filings related to business segments, products, and revenue:

\"\"\"{filing_text[:8000]}\"\"\"


Write a markdown answer with:

1. **Main Business Segments / Product Lines** – bullet list with 1–2 lines each.
2. **Which Segments Drive Growth vs. Profitability** – clearly distinguish, using hints from the text and metrics.
3. **Key Dependencies / Concentrations** – e.g., reliance on a platform, geography, or customer group.
4. **Product / Segment Risks** – 3–5 bullets.

Keep it focused on {analyzed_ticker}. Do NOT discuss other companies.
"""

        llm_resp = self.llm.invoke(prompt)
        return f"## 📦 Product & Segment Analysis for {analyzed_ticker}\n\n" + llm_resp.content.strip()


# ============================================================================
# PEER COMPARISON AGENT
# ============================================================================

class PeerComparisonAgent:
    """Peer comparison with head-to-head support"""

    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm

    def analyze(self, query: str) -> str:
        ticker1, ticker2 = extract_tickers_from_query(query, self.ticker)

        if ticker2:
            return self.head_to_head_comparison(ticker1, ticker2, query)
        else:
            return self.sector_comparison(ticker1, query)

    def head_to_head_comparison(self, ticker1: str, ticker2: str, query: str) -> str:
        """Direct comparison between two tickers"""

        metrics1 = json.loads(get_stock_metrics_tool(ticker1))
        metrics2 = json.loads(get_stock_metrics_tool(ticker2))

        # Health scores (best-effort)
        try:
            health1 = json.loads(calculate_health_score_tool(ticker1))
        except Exception:
            health1 = {}
        try:
            health2 = json.loads(calculate_health_score_tool(ticker2))
        except Exception:
            health2 = {}

        table = f"""## 🏆 {ticker1} vs {ticker2} – Peer Comparison

### Core Metrics

| Metric | {ticker1} | {ticker2} |
|--------|-----------|-----------|
| **Price** | ${metrics1.get('price', '–')} | ${metrics2.get('price', '–')} |
| **P/E** | {metrics1.get('pe_ratio', '–')} | {metrics2.get('pe_ratio', '–')} |
| **Revenue (B)** | {metrics1.get('revenue_b', '–')} | {metrics2.get('revenue_b', '–')} |
| **Rev. Growth %** | {metrics1.get('revenue_growth_pct', '–')} | {metrics2.get('revenue_growth_pct', '–')} |
| **Profit Margin %** | {metrics1.get('profit_margin_pct', '–')} | {metrics2.get('profit_margin_pct', '–')} |
| **Debt/Equity** | {metrics1.get('debt_to_equity', '–')} | {metrics2.get('debt_to_equity', '–')} |
| **Beta** | {metrics1.get('beta', '–')} | {metrics2.get('beta', '–')} |
| **Health Score** | {health1.get('score', '–')} | {health2.get('score', '–')} |
"""

        prompt = f"""You are comparing two large-cap stocks.

User question:
\"\"\"{query}\"\"\"


Metrics for {ticker1}:
{json.dumps(metrics1, indent=2)}

Health for {ticker1}: {json.dumps(health1, indent=2)}

Metrics for {ticker2}:
{json.dumps(metrics2, indent=2)}

Health for {ticker2}: {json.dumps(health2, indent=2)}

Write a concise markdown summary with:

1. **Valuation & Growth** – who looks cheaper / faster-growing.
2. **Profitability & Quality** – margins, balance sheet strength.
3. **Risk Profile** – volatility, leverage.
4. **Who Better Fits a {self.portfolio_config['risk_tolerance']} Investor** – high level, no explicit investment advice.

Be neutral and explain trade-offs instead of picking a “winner”.
"""

        llm_resp = self.llm.invoke(prompt)
        return table + "\n\n---\n\n" + llm_resp.content.strip()

    def sector_comparison(self, ticker: str, query: str) -> str:
        """Compare a ticker vs its sector peers using backup peer data where available"""

        peer_json = get_peer_comparison_tool(ticker)
        data = json.loads(peer_json)

        if 'error' in data:
            # Fallback: just ask LLM to describe positioning using metrics
            metrics = json.loads(get_stock_metrics_tool(ticker))
            prompt = f"""The user asked:
\"\"\"{query}\"\"\"


Here are metrics for {ticker}:
{json.dumps(metrics, indent=2)}

Explain in markdown how {ticker} compares to a typical company in its sector
in terms of growth, profitability, and risk. You don't have exact peer numbers,
so speak qualitatively based on the absolute metrics.
"""
            llm_resp = self.llm.invoke(prompt)
            return f"## 🏆 Peer & Sector Positioning for {ticker}\n\n" + llm_resp.content.strip()

        # Build peer table
        rows = []
        for peer in data.get('peers', []):
            rows.append(
                f"| {peer['ticker']} | {peer['pe']} | {peer['profit_margin_pct']}% | {peer['revenue_growth_pct']}% |"
            )

        peer_table = f"""## 🏆 Peer Comparison for {ticker}

### Sector: {data.get('sector', 'Unknown')}

#### {ticker} vs Sector Averages

| Metric | {ticker} | Sector Avg |
|--------|----------|-----------|
| **P/E** | {data.get('ticker_pe', '–')} | {data.get('sector_avg_pe', '–')} |
| **Profit Margin %** | {data.get('ticker_margin_pct', '–')} | {data.get('sector_avg_margin_pct', '–')} |
| **Revenue Growth %** | {data.get('ticker_growth_pct', '–')} | {data.get('sector_avg_growth_pct', '–')} |

#### Selected Peers

| Peer | P/E | Profit Margin % | Revenue Growth % |
|------|-----|-----------------|------------------|
{chr(10).join(rows)}
"""

        prompt = f"""User question:
\"\"\"{query}\"\"\"


Here is structured peer data for {ticker}:
{json.dumps(data, indent=2)}

Explain in markdown:

1. How {ticker} compares to its sector on valuation (P/E), profitability, and growth.
2. How it stacks up against the listed peers.
3. What this positioning implies for a {self.portfolio_config['risk_tolerance']} investor.

Be specific but concise.
"""

        llm_resp = self.llm.invoke(prompt)
        return peer_table + "\n\n---\n\n" + llm_resp.content.strip()


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

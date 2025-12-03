
"""
Multi-Agent System using Fireworks Llama 3.1 70B
Each agent specializes in a specific type of financial analysis
"""

import streamlit as st
from langchain_fireworks import ChatFireworks
from langchain.prompts import ChatPromptTemplate
from typing import Dict
from utils.tools import (
    get_stock_metrics_tool,
    search_sec_filing_tool,
    calculate_price_targets_tool,
    get_peer_comparison_tool,
    calculate_health_score_tool
)


@st.cache_resource
def get_llm():
    """Initialize Fireworks Llama 3.1 70B LLM"""
    llm = ChatFireworks(
        model="accounts/fireworks/models/llama-v3p3-70b-instruct",
        api_key=st.secrets["FIREWORKS_API_KEY"],
        temperature=0.3,
        max_tokens=2048
    )
    return llm


class FinancialAgentOrchestrator:
    """
    Orchestrator that routes queries to specialized agents
    Determines intent and delegates to appropriate specialist
    """
    
    def __init__(self, ticker: str, portfolio_config: Dict):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = get_llm()
    
    def classify_intent(self, query: str) -> str:
        """
        Classify user query to determine which agent should handle it
        
        Returns: 'investment', 'risk', 'product', 'comparison', or 'general'
        """
        query_lower = query.lower()
        
        # Simple keyword-based classification
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
        """Route query to appropriate specialized agent"""
        
        intent = self.classify_intent(query)
        
        # Route to appropriate agent
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
    """Specialized agent for investment decisions and recommendations"""
    
    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm
    
    def analyze(self, query: str) -> str:
        """Provide investment recommendation with specific entry/exit points"""
        
        with st.status("💼 Investment Advisor Agent", expanded=True) as status:
            st.write("🔧 Gathering financial metrics...")
            metrics = get_stock_metrics_tool(self.ticker)
            
            st.write("🎯 Calculating price targets...")
            targets = calculate_price_targets_tool(self.ticker, self.portfolio_config['risk_tolerance'])
            
            st.write("🏥 Assessing financial health...")
            health = calculate_health_score_tool(self.ticker)
            
            st.write("📄 Checking SEC filing for risks...")
            risks = search_sec_filing_tool(self.ticker, "risks concerns challenges")
            
            st.write("🤖 Synthesizing investment recommendation...")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a senior financial advisor providing investment recommendations.

RULES:
- Use ONLY the data provided below
- Include specific numbers and prices
- Be clear about BUY/HOLD/WAIT recommendation
- Explain your reasoning
- If data is missing, state it explicitly"""),
                ("human", """Provide investment recommendation for {ticker}.

USER PROFILE:
- Risk Tolerance: {risk_tolerance}
- Investment Horizon: {horizon}
- Target Allocation: {allocation}%

FINANCIAL METRICS:
{metrics}

PRICE TARGETS:
{targets}

HEALTH ASSESSMENT:
{health}

RISK FACTORS:
{risks}

USER QUESTION: {query}

Provide recommendation in this format:

## 💡 Investment Recommendation for {ticker}

**Verdict:** [BUY / HOLD / WAIT] - [one sentence why]

### 📊 Current Analysis
- Current Price: $[from targets]
- Fair Value: $[from targets]
- Health Score: [from health]

### 🎯 Price Targets ({risk_tolerance} profile)
- Entry Point: $[specific price]
- Target Exit: $[specific price]
- Stop Loss: $[specific price]

### ⚠️ Key Considerations
[List 2-3 most important factors from data]

### �� Action Plan
[Specific steps to take]

Use exact numbers from the data provided.""")
            ])
            
            messages = prompt.format_messages(
                ticker=self.ticker,
                risk_tolerance=self.portfolio_config['risk_tolerance'],
                horizon=self.portfolio_config['investment_horizon'],
                allocation=self.portfolio_config['portfolio_allocation'],
                metrics=metrics,
                targets=targets,
                health=health,
                risks=risks,
                query=query
            )
            
            response = self.llm.invoke(messages)
            status.update(label="✅ Investment Advisor Complete", state="complete")
            
        return response.content


class RiskAnalysisAgent:
    """Specialized agent for risk assessment and analysis"""
    
    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm
    
    def analyze(self, query: str) -> str:
        """Comprehensive risk analysis"""
        
        with st.status("🛡️ Risk Analysis Agent", expanded=True) as status:
            st.write("🔧 Gathering financial metrics...")
            metrics = get_stock_metrics_tool(self.ticker)
            
            st.write("📄 Extracting risk disclosures from SEC filing...")
            risks = search_sec_filing_tool(self.ticker, "risk factors threats concerns regulatory operational financial")
            
            st.write("🤖 Analyzing risk profile...")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a risk assessment specialist. Identify and categorize all investment risks clearly."),
                ("human", """Analyze investment risks for {ticker}.

FINANCIAL METRICS:
{metrics}

SEC RISK DISCLOSURES:
{risks}

USER QUESTION: {query}

Provide comprehensive risk analysis:

## 🛡️ Risk Assessment for {ticker}

### Financial Risks
[Analyze leverage, liquidity, margins from metrics]

### Business & Operational Risks
[Extract from SEC filing disclosures]

### Market & Volatility Risks
[Discuss beta, sector exposure]

### Overall Risk Rating
**Rating:** [Low / Medium / High]
**Rationale:** [Explain based on data]

### Risk Mitigation Strategies
[Suggest 2-3 specific ways to manage risks]

Be specific and cite numbers from the data.""")
            ])
            
            messages = prompt.format_messages(
                ticker=self.ticker,
                metrics=metrics,
                risks=risks,
                query=query
            )
            
            response = self.llm.invoke(messages)
            status.update(label="✅ Risk Analysis Complete", state="complete")
            
        return response.content


class ProductAnalysisAgent:
    """Specialized agent for product/segment performance analysis"""
    
    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm
    
    def analyze(self, query: str) -> str:
        """Analyze product lines and business segments"""
        
        with st.status("📦 Product Analysis Agent", expanded=True) as status:
            st.write("🔧 Gathering company metrics...")
            metrics = get_stock_metrics_tool(self.ticker)
            
            st.write("📄 Searching SEC filing for product/segment data...")
            products = search_sec_filing_tool(self.ticker, "product segment revenue sales performance growth business line")
            
            st.write("🤖 Analyzing product portfolio...")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a business analyst specializing in product and segment analysis."),
                ("human", """Analyze product/segment performance for {ticker}.

COMPANY METRICS:
{metrics}

PRODUCT/SEGMENT DATA FROM SEC FILING:
{products}

USER QUESTION: {query}

Provide analysis:

## 📦 Product Analysis for {ticker}

### Product/Segment Breakdown
[List products/segments mentioned with performance data]

### Performance Trends
[Which are growing? Which are declining?]

### Strategic Implications
[What does this mean for the company?]

### Investment Impact
[How should this affect investment decision?]

If specific product data is limited, analyze what overall metrics suggest.""")
            ])
            
            messages = prompt.format_messages(
                ticker=self.ticker,
                metrics=metrics,
                products=products,
                query=query
            )
            
            response = self.llm.invoke(messages)
            status.update(label="✅ Product Analysis Complete", state="complete")
            
        return response.content


class PeerComparisonAgent:
    """Specialized agent for competitive and peer analysis"""
    
    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm
    
    def analyze(self, query: str) -> str:
        """Compare company to sector peers"""
        
        with st.status("🏆 Peer Comparison Agent", expanded=True) as status:
            st.write("🔧 Gathering company metrics...")
            metrics = get_stock_metrics_tool(self.ticker)
            
            st.write("🏆 Fetching peer comparison data...")
            comparison = get_peer_comparison_tool(self.ticker)
            
            st.write("🤖 Analyzing competitive position...")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a competitive analyst specializing in sector comparisons."),
                ("human", """Compare {ticker} to sector peers.

{ticker} METRICS:
{metrics}

PEER COMPARISON:
{comparison}

USER QUESTION: {query}

Provide comparison:

## 🏆 Competitive Analysis for {ticker}

### Sector Positioning
[Where does {ticker} rank?]

### Valuation vs Peers
[P/E comparison - cheaper/expensive?]

### Profitability vs Peers
[Margin comparison - more/less profitable?]

### Growth vs Peers
[Revenue growth comparison - faster/slower?]

### Competitive Advantages
[What makes {ticker} unique?]

### Investment Verdict
[Better/worse pick vs peers?]

Use exact numbers to support all claims.""")
            ])
            
            messages = prompt.format_messages(
                ticker=self.ticker,
                metrics=metrics,
                comparison=comparison,
                query=query
            )
            
            response = self.llm.invoke(messages)
            status.update(label="✅ Peer Comparison Complete", state="complete")
            
        return response.content


class GeneralAnalysisAgent:
    """General purpose financial analysis agent"""
    
    def __init__(self, ticker: str, portfolio_config: Dict, llm):
        self.ticker = ticker
        self.portfolio_config = portfolio_config
        self.llm = llm
    
    def analyze(self, query: str) -> str:
        """General financial analysis"""
        
        with st.status("📊 General Analysis Agent", expanded=True) as status:
            st.write("🔧 Gathering financial data...")
            metrics = get_stock_metrics_tool(self.ticker)
            health = calculate_health_score_tool(self.ticker)
            filing = search_sec_filing_tool(self.ticker, query)
            
            st.write("🤖 Analyzing...")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a financial analyst providing comprehensive stock analysis."),
                ("human", """Analyze {ticker} to answer the user's question.

FINANCIAL METRICS:
{metrics}

HEALTH SCORE:
{health}

FROM SEC FILING:
{filing}

USER QUESTION: {query}

Provide detailed, data-driven analysis. Use specific numbers and cite relevant information.""")
            ])
            
            messages = prompt.format_messages(
                ticker=self.ticker,
                metrics=metrics,
                health=health,
                filing=filing,
                query=query
            )
            
            response = self.llm.invoke(messages)
            status.update(label="✅ Analysis Complete", state="complete")
            
        return response.content

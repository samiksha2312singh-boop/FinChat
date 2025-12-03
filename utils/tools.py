"""
Tool Functions for Multi-Agent System
These are deterministic operations, not AI agents
"""

import json
from typing import Dict

# ============================================================================
# SAMPLE DATA (Use this while Yahoo Finance is rate limited)
# ============================================================================

SAMPLE_METRICS = {
    'AAPL': {
        'ticker': 'AAPL',
        'company_name': 'Apple Inc.',
        'price': 175.43,
        'pe_ratio': 28.5,
        'forward_pe': 26.2,
        'market_cap_b': 2750.0,
        'revenue_b': 383.0,
        'revenue_growth_pct': 8.1,
        'profit_margin_pct': 25.3,
        'operating_margin_pct': 30.1,
        'debt_to_equity': 140.0,
        'current_ratio': 0.98,
        'beta': 1.24,
        'sector': 'Technology',
        'industry': 'Consumer Electronics'
    },
    'META': {
        'ticker': 'META',
        'company_name': 'Meta Platforms Inc.',
        'price': 338.54,
        'pe_ratio': 24.8,
        'forward_pe': 22.1,
        'market_cap_b': 865.0,
        'revenue_b': 134.9,
        'revenue_growth_pct': 22.1,
        'profit_margin_pct': 29.3,
        'operating_margin_pct': 38.6,
        'debt_to_equity': 0.0,
        'current_ratio': 2.86,
        'beta': 1.18,
        'sector': 'Communication Services',
        'industry': 'Internet Content & Information'
    },
    'MSFT': {
        'ticker': 'MSFT',
        'company_name': 'Microsoft Corporation',
        'price': 378.91,
        'pe_ratio': 35.2,
        'forward_pe': 31.8,
        'market_cap_b': 2820.0,
        'revenue_b': 211.0,
        'revenue_growth_pct': 12.5,
        'profit_margin_pct': 36.7,
        'operating_margin_pct': 42.0,
        'debt_to_equity': 52.0,
        'current_ratio': 1.77,
        'beta': 0.89,
        'sector': 'Technology',
        'industry': 'Software - Infrastructure'
    },
    'TSLA': {
        'ticker': 'TSLA',
        'company_name': 'Tesla Inc.',
        'price': 242.84,
        'pe_ratio': 77.3,
        'forward_pe': 68.5,
        'market_cap_b': 770.0,
        'revenue_b': 96.8,
        'revenue_growth_pct': 18.8,
        'profit_margin_pct': 14.5,
        'operating_margin_pct': 9.2,
        'debt_to_equity': 9.3,
        'current_ratio': 1.73,
        'beta': 2.39,
        'sector': 'Consumer Cyclical',
        'industry': 'Auto Manufacturers'
    },
    'GOOGL': {
        'ticker': 'GOOGL',
        'company_name': 'Alphabet Inc.',
        'price': 139.62,
        'pe_ratio': 23.1,
        'forward_pe': 20.8,
        'market_cap_b': 1750.0,
        'revenue_b': 307.4,
        'revenue_growth_pct': 8.7,
        'profit_margin_pct': 26.7,
        'operating_margin_pct': 30.2,
        'debt_to_equity': 6.8,
        'current_ratio': 2.93,
        'beta': 1.05,
        'sector': 'Communication Services',
        'industry': 'Internet Content & Information'
    },
    'NVDA': {
        'ticker': 'NVDA',
        'company_name': 'NVIDIA Corporation',
        'price': 487.23,
        'pe_ratio': 52.3,
        'forward_pe': 31.2,
        'market_cap_b': 1200.0,
        'revenue_b': 60.9,
        'revenue_growth_pct': 122.4,
        'profit_margin_pct': 48.9,
        'operating_margin_pct': 54.0,
        'debt_to_equity': 31.5,
        'current_ratio': 3.95,
        'beta': 1.68,
        'sector': 'Technology',
        'industry': 'Semiconductors'
    },
    'AMZN': {
        'ticker': 'AMZN',
        'company_name': 'Amazon.com Inc.',
        'price': 145.83,
        'pe_ratio': 44.2,
        'forward_pe': 28.7,
        'market_cap_b': 1510.0,
        'revenue_b': 574.8,
        'revenue_growth_pct': 12.2,
        'profit_margin_pct': 6.3,
        'operating_margin_pct': 8.2,
        'debt_to_equity': 78.5,
        'current_ratio': 1.09,
        'beta': 1.15,
        'sector': 'Consumer Cyclical',
        'industry': 'Internet Retail'
    }
}


# ============================================================================
# TOOL FUNCTIONS
# ============================================================================

def get_stock_metrics_tool(ticker: str) -> str:
    """
    Fetch stock metrics - USES SAMPLE DATA for demo
    (Yahoo Finance is rate limited)
    """
    
    # Use sample data if available
    if ticker in SAMPLE_METRICS:
        data = SAMPLE_METRICS[ticker].copy()
        data['data_source'] = '📊 Demo Data (Yahoo Finance rate limited - realistic values for testing)'
        return json.dumps(data, indent=2)
    
    # For other tickers, try Yahoo Finance
    try:
        import yfinance as yf
        import time
        time.sleep(2)  # Wait to avoid rate limit
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if info and len(info) > 10:
            metrics = {
                'ticker': ticker,
                'company_name': info.get('longName', ticker),
                'price': round(float(info.get('currentPrice', 0) or 0), 2),
                'pe_ratio': round(float(info.get('trailingPE', 0) or 0), 2),
                'forward_pe': round(float(info.get('forwardPE', 0) or 0), 2),
                'market_cap_b': round(float(info.get('marketCap', 0) or 0) / 1e9, 2),
                'revenue_b': round(float(info.get('totalRevenue', 0) or 0) / 1e9, 2),
                'revenue_growth_pct': round(float(info.get('revenueGrowth', 0) or 0) * 100, 1),
                'profit_margin_pct': round(float(info.get('profitMargins', 0) or 0) * 100, 1),
                'operating_margin_pct': round(float(info.get('operatingMargins', 0) or 0) * 100, 1),
                'debt_to_equity': round(float(info.get('debtToEquity', 0) or 0), 2),
                'current_ratio': round(float(info.get('currentRatio', 0) or 0), 2),
                'beta': round(float(info.get('beta', 1.0) or 1.0), 2),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'data_source': 'Live Yahoo Finance'
            }
            return json.dumps(metrics, indent=2)
    except:
        pass
    
    # No data available
    return json.dumps({
        'error': f'No data for {ticker}. Available demo tickers: {", ".join(SAMPLE_METRICS.keys())}',
        'ticker': ticker
    })


def search_sec_filing_tool(ticker: str, query: str) -> str:
    """
    Search SEC filing using RAG system
    """
    try:
        from utils.rag import search_filing
        
        results = search_filing(ticker, query, n_results=3)
        
        if not results:
            return f"SEC filing for {ticker} is uploaded but no content matched the query. Try broader search terms or the filing may still be processing."
        
        excerpts = []
        for i, result in enumerate(results, 1):
            excerpt = f"**[Section: {result['section']}]**\n{result['text'][:600]}\n"
            excerpts.append(excerpt)
        
        return "\n---\n".join(excerpts)
        
    except Exception as e:
        return f"RAG search error: {str(e)}. Make sure filing was uploaded and processed successfully."


def calculate_price_targets_tool(ticker: str, risk_tolerance: str) -> str:
    """
    Calculate fair value and price targets
    """
    
    # Get metrics (from sample or live data)
    if ticker in SAMPLE_METRICS:
        data = SAMPLE_METRICS[ticker]
        current_price = data['price']
        pe = data['pe_ratio']
        growth = data['revenue_growth_pct'] / 100
        margin = data['profit_margin_pct'] / 100
    else:
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            info = stock.info
            
            current_price = info.get('currentPrice', 0)
            pe = info.get('trailingPE', 20)
            growth = info.get('revenueGrowth', 0.05)
            margin = info.get('profitMargins', 0.10)
            
            if not current_price or current_price == 0:
                raise Exception("No price")
        except:
            return json.dumps({'error': f'No data available for {ticker}'})
    
    # Calculate fair value P/E based on growth
    if growth > 0.20:
        fair_pe = 35
    elif growth > 0.15:
        fair_pe = 30
    elif growth > 0.10:
        fair_pe = 25
    elif growth > 0.05:
        fair_pe = 20
    else:
        fair_pe = 15
    
    # Adjust for margin quality
    if margin > 0.25:
        fair_pe *= 1.1
    elif margin < 0.10:
        fair_pe *= 0.9
    
    # Calculate fair value
    eps = current_price / pe if pe > 0 else current_price / 20
    fair_value = eps * fair_pe
    
    # Risk-adjusted multipliers
    multipliers = {
        'Conservative': 0.85,
        'Moderate': 1.0,
        'Aggressive': 1.15
    }
    mult = multipliers.get(risk_tolerance, 1.0)
    
    # Calculate all price targets
    targets = {
        'ticker': ticker,
        'current_price': round(current_price, 2),
        'fair_value': round(fair_value, 2),
        'conservative_entry': round(fair_value * 0.85 * mult, 2),
        'moderate_entry': round(fair_value * 0.95 * mult, 2),
        'aggressive_entry': round(fair_value * 1.05 * mult, 2),
        'target_exit': round(fair_value * 1.15, 2),
        'stop_loss': round(current_price * 0.92, 2),
        'upside_percent': round(((fair_value - current_price) / current_price * 100) if current_price > 0 else 0, 1),
        'risk_tolerance': risk_tolerance
    }
    
    return json.dumps(targets, indent=2)


def get_peer_comparison_tool(ticker: str) -> str:
    """
    Compare stock to sector peers - SAMPLE DATA for demo
    """
    
    # Peer comparison sample data
    peer_comparisons = {
        'AAPL': {
            'ticker': 'AAPL',
            'sector': 'Technology',
            'ticker_pe': 28.5,
            'ticker_margin_pct': 25.3,
            'ticker_growth_pct': 8.1,
            'sector_avg_pe': 32.4,
            'sector_avg_margin_pct': 28.1,
            'sector_avg_growth_pct': 13.2,
            'peers': [
                {'ticker': 'MSFT', 'pe': 35.2, 'profit_margin_pct': 36.7, 'revenue_growth_pct': 12.5},
                {'ticker': 'GOOGL', 'pe': 23.1, 'profit_margin_pct': 26.7, 'revenue_growth_pct': 8.7},
                {'ticker': 'META', 'pe': 24.8, 'profit_margin_pct': 29.3, 'revenue_growth_pct': 22.1},
                {'ticker': 'NVDA', 'pe': 52.3, 'profit_margin_pct': 48.9, 'revenue_growth_pct': 122.4}
            ]
        },
        'META': {
            'ticker': 'META',
            'sector': 'Communication Services',
            'ticker_pe': 24.8,
            'ticker_margin_pct': 29.3,
            'ticker_growth_pct': 22.1,
            'sector_avg_pe': 26.8,
            'sector_avg_margin_pct': 24.5,
            'sector_avg_growth_pct': 10.3,
            'peers': [
                {'ticker': 'GOOGL', 'pe': 23.1, 'profit_margin_pct': 26.7, 'revenue_growth_pct': 8.7},
                {'ticker': 'NFLX', 'pe': 38.5, 'profit_margin_pct': 16.2, 'revenue_growth_pct': 6.5},
                {'ticker': 'DIS', 'pe': 45.2, 'profit_margin_pct': 7.8, 'revenue_growth_pct': 5.2}
            ]
        },
        'MSFT': {
            'ticker': 'MSFT',
            'sector': 'Technology',
            'ticker_pe': 35.2,
            'ticker_margin_pct': 36.7,
            'ticker_growth_pct': 12.5,
            'sector_avg_pe': 32.4,
            'sector_avg_margin_pct': 28.1,
            'sector_avg_growth_pct': 13.2,
            'peers': [
                {'ticker': 'AAPL', 'pe': 28.5, 'profit_margin_pct': 25.3, 'revenue_growth_pct': 8.1},
                {'ticker': 'GOOGL', 'pe': 23.1, 'profit_margin_pct': 26.7, 'revenue_growth_pct': 8.7},
                {'ticker': 'ORCL', 'pe': 35.8, 'profit_margin_pct': 22.1, 'revenue_growth_pct': 6.8}
            ]
        },
        'TSLA': {
            'ticker': 'TSLA',
            'sector': 'Consumer Cyclical',
            'ticker_pe': 77.3,
            'ticker_margin_pct': 14.5,
            'ticker_growth_pct': 18.8,
            'sector_avg_pe': 24.2,
            'sector_avg_margin_pct': 8.5,
            'sector_avg_growth_pct': 7.2,
            'peers': [
                {'ticker': 'GM', 'pe': 5.8, 'profit_margin_pct': 5.2, 'revenue_growth_pct': 2.1},
                {'ticker': 'F', 'pe': 6.2, 'profit_margin_pct': 3.8, 'revenue_growth_pct': 1.5},
                {'ticker': 'RIVN', 'pe': 0, 'profit_margin_pct': -58.2, 'revenue_growth_pct': 167.3}
            ]
        },
        'GOOGL': {
            'ticker': 'GOOGL',
            'sector': 'Communication Services',
            'ticker_pe': 23.1,
            'ticker_margin_pct': 26.7,
            'ticker_growth_pct': 8.7,
            'sector_avg_pe': 26.8,
            'sector_avg_margin_pct': 24.5,
            'sector_avg_growth_pct': 10.3,
            'peers': [
                {'ticker': 'META', 'pe': 24.8, 'profit_margin_pct': 29.3, 'revenue_growth_pct': 22.1},
                {'ticker': 'NFLX', 'pe': 38.5, 'profit_margin_pct': 16.2, 'revenue_growth_pct': 6.5}
            ]
        }
    }
    
    if ticker in peer_comparisons:
        return json.dumps(peer_comparisons[ticker], indent=2)
    else:
        return json.dumps({
            'error': f'Peer comparison sample data only for: {", ".join(peer_comparisons.keys())}',
            'ticker': ticker
        })


def calculate_health_score_tool(ticker: str) -> str:
    """
    Calculate financial health score using sample or live data
    """
    
    # Get metrics
    if ticker in SAMPLE_METRICS:
        data = SAMPLE_METRICS[ticker]
        margin = data['profit_margin_pct'] / 100
        dte = data['debt_to_equity']
        growth = data['revenue_growth_pct'] / 100
        pe = data['pe_ratio']
    else:
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            info = stock.info
            
            margin = info.get('profitMargins', 0)
            dte = info.get('debtToEquity', 0)
            growth = info.get('revenueGrowth', 0)
            pe = info.get('trailingPE', 0)
        except:
            return json.dumps({'error': f'No data for {ticker}'})
    
    # Calculate score
    score = 50
    factors = []
    
    # Profitability
    if margin > 0.15:
        score += 15
        factors.append("✅ Strong profit margins (>15%)")
    elif margin > 0.05:
        score += 5
        factors.append("⚠️ Moderate profit margins (5-15%)")
    else:
        score -= 10
        factors.append("❌ Low profit margins (<5%)")
    
    # Leverage
    if dte and dte < 50:
        score += 10
        factors.append("✅ Conservative debt (D/E < 50)")
    elif dte > 150:
        score -= 15
        factors.append("❌ High leverage (D/E > 150)")
    elif dte > 0:
        factors.append("⚠️ Moderate debt levels")
    
    # Growth
    if growth > 0.15:
        score += 15
        factors.append("✅ Strong revenue growth (>15%)")
    elif growth < 0:
        score -= 10
        factors.append("❌ Revenue contraction")
    else:
        factors.append("⚠️ Moderate growth")
    
    # Valuation
    if pe and 10 < pe < 25:
        score += 10
        factors.append("✅ Reasonable valuation (P/E 10-25)")
    elif pe > 50:
        score -= 5
        factors.append("⚠️ High valuation (P/E > 50)")
    
    score = max(0, min(100, score))
    
    # Rating
    if score > 80:
        rating = "Excellent"
    elif score > 60:
        rating = "Good"
    elif score > 40:
        rating = "Fair"
    else:
        rating = "Poor"
    
    return json.dumps({
        'ticker': ticker,
        'score': score,
        'rating': rating,
        'factors': factors
    }, indent=2)
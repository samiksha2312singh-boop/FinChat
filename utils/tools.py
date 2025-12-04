"""
Tool Functions for Multi-Agent System
HYBRID: Tries live data first, falls back to current backup data
UPDATED: Includes RAG with reranking for better SEC filing search
"""

import json
from typing import Dict

# ============================================================================
# BACKUP DATA - Current as of December 2024
# ============================================================================

BACKUP_METRICS = {
    'AAPL': {
        'ticker': 'AAPL',
        'company_name': 'Apple Inc.',
        'price': 284.15,
        'pe_ratio': 45.8,
        'forward_pe': 35.2,
        'market_cap_b': 4350.0,
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
    'MSFT': {
        'ticker': 'MSFT',
        'company_name': 'Microsoft Corporation',
        'price': 477.73,
        'pe_ratio': 35.2,
        'forward_pe': 31.8,
        'market_cap_b': 3100.0,
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
    'META': {
        'ticker': 'META',
        'company_name': 'Meta Platforms Inc.',
        'price': 639.60,
        'pe_ratio': 28.5,
        'forward_pe': 24.1,
        'market_cap_b': 1480.0,
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
    'GOOGL': {
        'ticker': 'GOOGL',
        'company_name': 'Alphabet Inc.',
        'price': 320.62,
        'pe_ratio': 23.1,
        'forward_pe': 20.8,
        'market_cap_b': 2150.0,
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
        'price': 179.59,
        'pe_ratio': 52.3,
        'forward_pe': 31.2,
        'market_cap_b': 3450.0,
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
    'TSLA': {
        'ticker': 'TSLA',
        'company_name': 'Tesla Inc.',
        'price': 446.74,
        'pe_ratio': 77.3,
        'forward_pe': 68.5,
        'market_cap_b': 1120.0,
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
    'AMZN': {
        'ticker': 'AMZN',
        'company_name': 'Amazon.com Inc.',
        'price': 238.35,
        'pe_ratio': 44.2,
        'forward_pe': 28.7,
        'market_cap_b': 2240.0,
        'revenue_b': 574.8,
        'revenue_growth_pct': 12.2,
        'profit_margin_pct': 6.3,
        'operating_margin_pct': 8.2,
        'debt_to_equity': 78.5,
        'current_ratio': 1.09,
        'beta': 1.15,
        'sector': 'Consumer Cyclical',
        'industry': 'Internet Retail'
    },
    'V': {
        'ticker': 'V',
        'company_name': 'Visa Inc.',
        'price': 329.61,
        'pe_ratio': 33.5,
        'forward_pe': 28.9,
        'market_cap_b': 653.0,
        'revenue_b': 32.7,
        'revenue_growth_pct': 9.6,
        'profit_margin_pct': 51.2,
        'operating_margin_pct': 67.3,
        'debt_to_equity': 65.8,
        'current_ratio': 1.42,
        'beta': 0.98,
        'sector': 'Financial Services',
        'industry': 'Credit Services'
    },
    'JPM': {
        'ticker': 'JPM',
        'company_name': 'JPMorgan Chase & Co.',
        'price': 312.13,
        'pe_ratio': 12.8,
        'forward_pe': 11.5,
        'market_cap_b': 698.0,
        'revenue_b': 162.4,
        'revenue_growth_pct': 7.8,
        'profit_margin_pct': 28.5,
        'operating_margin_pct': 38.2,
        'debt_to_equity': 125.4,
        'current_ratio': 1.15,
        'beta': 1.12,
        'sector': 'Financial Services',
        'industry': 'Banks - Diversified'
    },
    'NFLX': {
        'ticker': 'NFLX',
        'company_name': 'Netflix Inc.',
        'price': 103.96,
        'pe_ratio': 48.6,
        'forward_pe': 35.2,
        'market_cap_b': 382.0,
        'revenue_b': 33.7,
        'revenue_growth_pct': 6.5,
        'profit_margin_pct': 16.2,
        'operating_margin_pct': 20.1,
        'debt_to_equity': 89.3,
        'current_ratio': 1.23,
        'beta': 1.34,
        'sector': 'Communication Services',
        'industry': 'Entertainment'
    },
    'DIS': {
        'ticker': 'DIS',
        'company_name': 'The Walt Disney Company',
        'price': 105.74,
        'pe_ratio': 45.2,
        'forward_pe': 28.6,
        'market_cap_b': 204.0,
        'revenue_b': 89.5,
        'revenue_growth_pct': 5.2,
        'profit_margin_pct': 7.8,
        'operating_margin_pct': 11.3,
        'debt_to_equity': 58.7,
        'current_ratio': 0.89,
        'beta': 1.23,
        'sector': 'Communication Services',
        'industry': 'Entertainment'
    }
}


# ============================================================================
# TOOL FUNCTIONS
# ============================================================================

def get_stock_metrics_tool(ticker: str) -> str:
    """
    Fetch stock metrics - HYBRID APPROACH
    Tries live Yahoo Finance first, falls back to backup
    """
    
    import yfinance as yf
    import time
    
    # Try live data
    try:
        time.sleep(0.5)
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        if current_price and current_price > 0 and len(info) > 10:
            # Live data available
            metrics = {
                'ticker': ticker.upper(),
                'company_name': info.get('longName', ticker),
                'price': round(float(current_price), 2),
                'pe_ratio': round(float(info.get('trailingPE') or 0), 2),
                'forward_pe': round(float(info.get('forwardPE') or 0), 2),
                'market_cap_b': round(float(info.get('marketCap') or 0) / 1e9, 2),
                'revenue_b': round(float(info.get('totalRevenue') or 0) / 1e9, 2),
                'revenue_growth_pct': round(float(info.get('revenueGrowth') or 0) * 100, 1),
                'profit_margin_pct': round(float(info.get('profitMargins') or 0) * 100, 1),
                'operating_margin_pct': round(float(info.get('operatingMargins') or 0) * 100, 1),
                'debt_to_equity': round(float(info.get('debtToEquity') or 0), 2),
                'current_ratio': round(float(info.get('currentRatio') or 0), 2),
                'beta': round(float(info.get('beta') or 1.0), 2),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'data_source': '✅ Live Yahoo Finance (Real-time)'
            }
            return json.dumps(metrics, indent=2)
    except:
        pass
    
    # Fallback to backup
    if ticker.upper() in BACKUP_METRICS:
        data = BACKUP_METRICS[ticker.upper()].copy()
        data['data_source'] = '⚠️ Backup Data (Yahoo Finance rate limited)'
        return json.dumps(data, indent=2)
    
    return json.dumps({
        'error': f'No data available for {ticker}',
        'ticker': ticker.upper(),
        'available_tickers': list(BACKUP_METRICS.keys())
    }, indent=2)


def search_sec_filing_tool(ticker: str, query: str) -> str:
    """
    Search SEC filing using RAG with LLM reranking
    """
    
    try:
        from utils.rag import rerank_chunks
        
        # Use reranking for better results
        results = rerank_chunks(ticker, query, n_results=3)
        
        if not results:
            return f"No SEC filing data for {ticker}. Upload 10-Q in sidebar."
        
        # Format with scores
        excerpts = []
        for i, result in enumerate(results, 1):
            score_info = f" (Relevance: {result.get('rerank_score', 0)}/10)" if 'rerank_score' in result else ""
            excerpt = f"**[{result['section']}]{score_info}**\n{result['text'][:700]}\n"
            excerpts.append(excerpt)
        
        return "\n---\n".join(excerpts)
        
    except Exception as e:
        return f"Error: {str(e)}"


def calculate_price_targets_tool(ticker: str, risk_tolerance: str) -> str:
    """Calculate fair value and price targets - HYBRID"""
    
    import yfinance as yf
    import time
    
    # Try live data
    try:
        time.sleep(0.5)
        stock = yf.Ticker(ticker)
        info = stock.info
        
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        if current_price and current_price > 0:
            pe = info.get('trailingPE', 20)
            growth = info.get('revenueGrowth', 0.05)
            margin = info.get('profitMargins', 0.10)
        else:
            raise Exception("Use backup")
    except:
        # Fallback
        if ticker.upper() in BACKUP_METRICS:
            data = BACKUP_METRICS[ticker.upper()]
            current_price = data['price']
            pe = data['pe_ratio']
            growth = data['revenue_growth_pct'] / 100
            margin = data['profit_margin_pct'] / 100
        else:
            return json.dumps({'error': f'No data for {ticker}'}, indent=2)
    
    # Fair value calculation
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
    
    if margin > 0.25:
        fair_pe *= 1.1
    elif margin < 0.10:
        fair_pe *= 0.9
    
    eps = current_price / pe if pe > 0 else current_price / 20
    fair_value = eps * fair_pe
    
    multipliers = {'Conservative': 0.85, 'Moderate': 1.0, 'Aggressive': 1.15}
    mult = multipliers.get(risk_tolerance, 1.0)
    
    targets = {
        'ticker': ticker.upper(),
        'current_price': round(current_price, 2),
        'fair_value': round(fair_value, 2),
        'conservative_entry': round(fair_value * 0.85 * mult, 2),
        'moderate_entry': round(fair_value * 0.95 * mult, 2),
        'aggressive_entry': round(fair_value * 1.05 * mult, 2),
        'target_exit': round(fair_value * 1.15, 2),
        'stop_loss': round(current_price * 0.92, 2),
        'upside_percent': round(((fair_value - current_price) / current_price * 100), 1),
        'risk_tolerance': risk_tolerance
    }
    
    return json.dumps(targets, indent=2)


def get_peer_comparison_tool(ticker: str) -> str:
    """Compare to sector peers - BACKUP DATA"""
    
    peer_comparisons = {
        'AAPL': {
            'ticker': 'AAPL',
            'sector': 'Technology',
            'ticker_pe': 45.8,
            'ticker_margin_pct': 25.3,
            'ticker_growth_pct': 8.1,
            'sector_avg_pe': 38.4,
            'sector_avg_margin_pct': 32.1,
            'sector_avg_growth_pct': 38.2,
            'peers': [
                {'ticker': 'MSFT', 'pe': 35.2, 'profit_margin_pct': 36.7, 'revenue_growth_pct': 12.5},
                {'ticker': 'GOOGL', 'pe': 23.1, 'profit_margin_pct': 26.7, 'revenue_growth_pct': 8.7},
                {'ticker': 'META', 'pe': 28.5, 'profit_margin_pct': 29.3, 'revenue_growth_pct': 22.1},
                {'ticker': 'NVDA', 'pe': 52.3, 'profit_margin_pct': 48.9, 'revenue_growth_pct': 122.4}
            ]
        },
        'META': {
            'ticker': 'META',
            'sector': 'Communication Services',
            'ticker_pe': 28.5,
            'ticker_margin_pct': 29.3,
            'ticker_growth_pct': 22.1,
            'sector_avg_pe': 26.8,
            'sector_avg_margin_pct': 24.5,
            'sector_avg_growth_pct': 10.3,
            'peers': [
                {'ticker': 'GOOGL', 'pe': 23.1, 'profit_margin_pct': 26.7, 'revenue_growth_pct': 8.7},
                {'ticker': 'NFLX', 'pe': 48.6, 'profit_margin_pct': 16.2, 'revenue_growth_pct': 6.5},
                {'ticker': 'DIS', 'pe': 45.2, 'profit_margin_pct': 7.8, 'revenue_growth_pct': 5.2}
            ]
        },
        'MSFT': {
            'ticker': 'MSFT',
            'sector': 'Technology',
            'ticker_pe': 35.2,
            'ticker_margin_pct': 36.7,
            'ticker_growth_pct': 12.5,
            'sector_avg_pe': 38.4,
            'sector_avg_margin_pct': 32.1,
            'sector_avg_growth_pct': 38.2,
            'peers': [
                {'ticker': 'AAPL', 'pe': 45.8, 'profit_margin_pct': 25.3, 'revenue_growth_pct': 8.1},
                {'ticker': 'GOOGL', 'pe': 23.1, 'profit_margin_pct': 26.7, 'revenue_growth_pct': 8.7}
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
                {'ticker': 'F', 'pe': 6.2, 'profit_margin_pct': 3.8, 'revenue_growth_pct': 1.5}
            ]
        },
        'NVDA': {
            'ticker': 'NVDA',
            'sector': 'Technology',
            'ticker_pe': 52.3,
            'ticker_margin_pct': 48.9,
            'ticker_growth_pct': 122.4,
            'sector_avg_pe': 38.4,
            'sector_avg_margin_pct': 32.1,
            'sector_avg_growth_pct': 38.2,
            'peers': [
                {'ticker': 'AMD', 'pe': 145.2, 'profit_margin_pct': 4.2, 'revenue_growth_pct': 8.9},
                {'ticker': 'INTC', 'pe': 28.5, 'profit_margin_pct': 12.4, 'revenue_growth_pct': -8.2}
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
                {'ticker': 'META', 'pe': 28.5, 'profit_margin_pct': 29.3, 'revenue_growth_pct': 22.1},
                {'ticker': 'NFLX', 'pe': 48.6, 'profit_margin_pct': 16.2, 'revenue_growth_pct': 6.5}
            ]
        }
    }
    
    if ticker.upper() in peer_comparisons:
        return json.dumps(peer_comparisons[ticker.upper()], indent=2)
    else:
        return json.dumps({
            'error': f'Peer data not available for {ticker}',
            'available': list(peer_comparisons.keys())
        }, indent=2)


def calculate_health_score_tool(ticker: str) -> str:
    """Calculate financial health score - HYBRID"""
    
    import yfinance as yf
    import time
    
    # Try live
    try:
        time.sleep(0.5)
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if info and len(info) > 10:
            margin = info.get('profitMargins', 0)
            dte = info.get('debtToEquity', 0)
            growth = info.get('revenueGrowth', 0)
            pe = info.get('trailingPE', 0)
        else:
            raise Exception("Use backup")
    except:
        # Fallback
        if ticker.upper() in BACKUP_METRICS:
            data = BACKUP_METRICS[ticker.upper()]
            margin = data['profit_margin_pct'] / 100
            dte = data['debt_to_equity']
            growth = data['revenue_growth_pct'] / 100
            pe = data['pe_ratio']
        else:
            return json.dumps({'error': f'No data for {ticker}'}, indent=2)
    
    # Calculate
    score = 50
    factors = []
    
    if margin > 0.15:
        score += 15
        factors.append("✅ Strong profit margins (>15%)")
    elif margin > 0.05:
        score += 5
        factors.append("⚠️ Moderate margins")
    else:
        score -= 10
        factors.append("❌ Low margins")
    
    if dte and dte < 50:
        score += 10
        factors.append("✅ Conservative debt")
    elif dte > 150:
        score -= 15
        factors.append("❌ High leverage")
    
    if growth > 0.15:
        score += 15
        factors.append("✅ Strong growth")
    elif growth < 0:
        score -= 10
        factors.append("❌ Revenue decline")
    
    if pe and 10 < pe < 25:
        score += 10
        factors.append("✅ Reasonable valuation")
    elif pe > 50:
        score -= 5
        factors.append("⚠️ High valuation")
    
    score = max(0, min(100, score))
    
    rating = 'Excellent' if score > 80 else 'Good' if score > 60 else 'Fair' if score > 40 else 'Poor'
    
    return json.dumps({
        'ticker': ticker.upper(),
        'score': score,
        'rating': rating,
        'factors': factors
    }, indent=2)
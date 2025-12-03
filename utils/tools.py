
"""
Tool Functions for Multi-Agent System
These are deterministic operations, not AI agents
"""

import yfinance as yf
import json
from typing import Dict
from utils.rag import search_filing


def get_stock_metrics_tool(ticker: str) -> str:
    """
    Fetch real-time stock metrics from Yahoo Finance
    Falls back to sample data if rate limited
    """
    try:
        import yfinance as yf
        import time
        
        # Add small delay to avoid rate limiting
        time.sleep(0.5)
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Check if we got real data
        if not info or len(info) < 5:
            raise Exception("Rate limited or no data")
        
        metrics = {
            'ticker': ticker,
            'company_name': info.get('longName', ticker),
            'price': round(float(info.get('currentPrice', 0) or info.get('regularMarketPrice', 0)), 2),
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
            'industry': info.get('industry', 'Unknown')
        }
        
        return json.dumps(metrics, indent=2)
        
    except Exception as e:
        # FALLBACK: Return sample data for common tickers
        sample_data = {
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
                'industry': 'Consumer Electronics',
                'note': 'SAMPLE DATA - Yahoo Finance rate limited'
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
                'industry': 'Software',
                'note': 'SAMPLE DATA - Yahoo Finance rate limited'
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
                'industry': 'Auto Manufacturers',
                'note': 'SAMPLE DATA - Yahoo Finance rate limited'
            }
        }
        
        if ticker in sample_data:
            return json.dumps(sample_data[ticker], indent=2)
        else:
            return json.dumps({
                'error': f'Yahoo Finance rate limited and no sample data for {ticker}',
                'ticker': ticker,
                'suggestion': 'Wait 10 minutes or use AAPL/MSFT/TSLA for demo'
            })
def get_stock_metrics_tool(ticker: str) -> str:
    """Fetch stock metrics with fallback sample data"""
    
    # SAMPLE DATA for demo (Yahoo Finance is rate limited)
    sample_data = {
        'AAPL': {
            'ticker': 'AAPL', 'company_name': 'Apple Inc.',
            'price': 175.43, 'pe_ratio': 28.5, 'forward_pe': 26.2,
            'market_cap_b': 2750.0, 'revenue_b': 383.0,
            'revenue_growth_pct': 8.1, 'profit_margin_pct': 25.3,
            'operating_margin_pct': 30.1, 'debt_to_equity': 140.0,
            'current_ratio': 0.98, 'beta': 1.24,
            'sector': 'Technology', 'industry': 'Consumer Electronics'
        },
        'META': {
            'ticker': 'META', 'company_name': 'Meta Platforms Inc.',
            'price': 338.54, 'pe_ratio': 24.8, 'forward_pe': 22.1,
            'market_cap_b': 865.0, 'revenue_b': 134.9,
            'revenue_growth_pct': 22.1, 'profit_margin_pct': 29.3,
            'operating_margin_pct': 38.6, 'debt_to_equity': 0.0,
            'current_ratio': 2.86, 'beta': 1.18,
            'sector': 'Communication Services', 'industry': 'Internet Content'
        },
        'MSFT': {
            'ticker': 'MSFT', 'company_name': 'Microsoft Corporation',
            'price': 378.91, 'pe_ratio': 35.2, 'forward_pe': 31.8,
            'market_cap_b': 2820.0, 'revenue_b': 211.0,
            'revenue_growth_pct': 12.5, 'profit_margin_pct': 36.7,
            'operating_margin_pct': 42.0, 'debt_to_equity': 52.0,
            'current_ratio': 1.77, 'beta': 0.89,
            'sector': 'Technology', 'industry': 'Software'
        },
        'TSLA': {
            'ticker': 'TSLA', 'company_name': 'Tesla Inc.',
            'price': 242.84, 'pe_ratio': 77.3, 'forward_pe': 68.5,
            'market_cap_b': 770.0, 'revenue_b': 96.8,
            'revenue_growth_pct': 18.8, 'profit_margin_pct': 14.5,
            'operating_margin_pct': 9.2, 'debt_to_equity': 9.3,
            'current_ratio': 1.73, 'beta': 2.39,
            'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers'
        },
        'GOOGL': {
            'ticker': 'GOOGL', 'company_name': 'Alphabet Inc.',
            'price': 139.62, 'pe_ratio': 23.1, 'forward_pe': 20.8,
            'market_cap_b': 1750.0, 'revenue_b': 307.4,
            'revenue_growth_pct': 8.7, 'profit_margin_pct': 26.7,
            'operating_margin_pct': 30.2, 'debt_to_equity': 6.8,
            'current_ratio': 2.93, 'beta': 1.05,
            'sector': 'Communication Services', 'industry': 'Internet Content'
        }
    }
    
    # Try Yahoo Finance first
    try:
        import yfinance as yf
        import time
        time.sleep(1)  # Avoid rate limit
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if info and len(info) > 10 and info.get('currentPrice'):
            # Real data available
            metrics = {
                'ticker': ticker,
                'company_name': info.get('longName', ticker),
                'price': round(float(info.get('currentPrice', 0)), 2),
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
                'industry': info.get('industry', 'Unknown')
            }
            return json.dumps(metrics, indent=2)
    except:
        pass  # Fall through to sample data
    
    # Use sample data
    if ticker in sample_data:
        sample_data[ticker]['data_source'] = 'SAMPLE DATA (Yahoo Finance rate limited)'
        return json.dumps(sample_data[ticker], indent=2)
    else:
        return json.dumps({
            'error': f'Yahoo Finance rate limited. Sample data only available for: {", ".join(sample_data.keys())}',
            'ticker': ticker
        })

def search_sec_filing_tool(ticker: str, query: str) -> str:
    """
    Search SEC filing using RAG system
    
    Args:
        ticker: Stock symbol
        query: Search query
    
    Returns:
        Formatted string with relevant filing excerpts
    """
    try:
        results = search_filing(ticker, query, n_results=3)
        
        if not results:
            return f"No SEC filing data available for {ticker}. Please upload a 10-Q filing first."
        
        excerpts = []
        for i, result in enumerate(results, 1):
            excerpt = f"**[{result['section']}]**\n{result['text']}\n"
            excerpts.append(excerpt)
        
        return "\n".join(excerpts)
        
    except Exception as e:
        return f"Error searching filing: {str(e)}"


def calculate_price_targets_tool(ticker: str, risk_tolerance: str) -> str:
    """Calculate price targets with fallback"""
    try:
        import yfinance as yf
        import time
        
        time.sleep(0.5)
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        pe = info.get('trailingPE', 20)
        growth = info.get('revenueGrowth', 0.05)
        margin = info.get('profitMargins', 0.10)
        
        if not current_price or current_price == 0:
            raise Exception("No price data")
        
    except:
        # Fallback prices for common tickers
        fallback = {
            'AAPL': (175.43, 28.5, 0.081, 0.253),
            'MSFT': (378.91, 35.2, 0.125, 0.367),
            'TSLA': (242.84, 77.3, 0.188, 0.145)
        }
        
        if ticker in fallback:
            current_price, pe, growth, margin = fallback[ticker]
        else:
            return json.dumps({'error': f'Rate limited and no fallback for {ticker}'})
    
    # Rest of calculation stays the same
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
        'ticker': ticker,
        'current_price': round(current_price, 2),
        'fair_value': round(fair_value, 2),
        'conservative_entry': round(fair_value * 0.85 * mult, 2),
        'moderate_entry': round(fair_value * 0.95 * mult, 2),
        'target_exit': round(fair_value * 1.15, 2),
        'stop_loss': round(current_price * 0.92, 2),
        'upside_percent': round(((fair_value - current_price) / current_price * 100) if current_price > 0 else 0, 1)
    }
    
    return json.dumps(targets, indent=2)
def calculate_price_targets_tool(ticker: str, risk_tolerance: str) -> str:
    """Calculate price targets with fallback"""
    
    # Fallback data
    fallback = {
        'AAPL': (175.43, 28.5, 0.081, 0.253),
        'META': (338.54, 24.8, 0.221, 0.293),
        'MSFT': (378.91, 35.2, 0.125, 0.367),
        'TSLA': (242.84, 77.3, 0.188, 0.145),
        'GOOGL': (139.62, 23.1, 0.087, 0.267)
    }
    
    # Try Yahoo Finance
    try:
        import yfinance as yf
        import time
        time.sleep(1)
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        pe = info.get('trailingPE', 20)
        growth = info.get('revenueGrowth', 0.05)
        margin = info.get('profitMargins', 0.10)
        
        if not current_price or current_price == 0:
            raise Exception("No price")
    except:
        # Use fallback
        if ticker in fallback:
            current_price, pe, growth, margin = fallback[ticker]
        else:
            return json.dumps({'error': f'No data for {ticker}'})
    
    # Calculate fair value (rest stays same)
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
        'ticker': ticker,
        'current_price': round(current_price, 2),
        'fair_value': round(fair_value, 2),
        'conservative_entry': round(fair_value * 0.85 * mult, 2),
        'moderate_entry': round(fair_value * 0.95 * mult, 2),
        'target_exit': round(fair_value * 1.15, 2),
        'stop_loss': round(current_price * 0.92, 2),
        'upside_percent': round(((fair_value - current_price) / current_price * 100) if current_price > 0 else 0, 1),
        'data_source': 'SAMPLE (Yahoo rate limited)' if ticker in fallback else 'LIVE'
    }
    
    return json.dumps(targets, indent=2)

def get_peer_comparison_tool(ticker: str) -> str:
    """
    Compare stock to sector peers
    
    Args:
        ticker: Stock symbol
    
    Returns:
        JSON string with peer comparison data
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get('sector', 'Technology')
        
        # Sector peer mapping
        peers_map = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
            'Consumer Cyclical': ['AMZN', 'TSLA', 'NKE', 'SBUX', 'HD'],
            'Communication Services': ['META', 'GOOGL', 'DIS', 'NFLX', 'T'],
            'Consumer Defensive': ['WMT', 'PG', 'KO', 'PEP', 'COST'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
        }
        
        peers = [p for p in peers_map.get(sector, ['SPY']) if p != ticker][:4]
        
        # Fetch peer data
        peer_data = []
        for peer_ticker in peers:
            try:
                peer = yf.Ticker(peer_ticker)
                peer_info = peer.info
                peer_data.append({
                    'ticker': peer_ticker,
                    'pe': round(peer_info.get('trailingPE', 0), 2),
                    'profit_margin_pct': round(peer_info.get('profitMargins', 0) * 100, 1),
                    'revenue_growth_pct': round(peer_info.get('revenueGrowth', 0) * 100, 1),
                    'debt_to_equity': round(peer_info.get('debtToEquity', 0), 2)
                })
            except:
                continue
        
        # Calculate sector averages
        if peer_data:
            avg_pe = sum(p['pe'] for p in peer_data if p['pe'] > 0) / len([p for p in peer_data if p['pe'] > 0]) if any(p['pe'] > 0 for p in peer_data) else 0
            avg_margin = sum(p['profit_margin_pct'] for p in peer_data) / len(peer_data)
            avg_growth = sum(p['revenue_growth_pct'] for p in peer_data) / len(peer_data)
        else:
            avg_pe = avg_margin = avg_growth = 0
        
        comparison = {
            'ticker': ticker,
            'sector': sector,
            'ticker_pe': round(info.get('trailingPE', 0), 2),
            'ticker_margin_pct': round(info.get('profitMargins', 0) * 100, 1),
            'ticker_growth_pct': round(info.get('revenueGrowth', 0) * 100, 1),
            'sector_avg_pe': round(avg_pe, 2),
            'sector_avg_margin_pct': round(avg_margin, 1),
            'sector_avg_growth_pct': round(avg_growth, 1),
            'peers': peer_data
        }
        
        return json.dumps(comparison, indent=2)
        
    except Exception as e:
        return json.dumps({'error': str(e)})


def calculate_health_score_tool(ticker: str) -> str:
    """
    Calculate financial health score (0-100)
    
    Args:
        ticker: Stock symbol
    
    Returns:
        JSON string with health score and factors
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        score = 50
        factors = []
        
        # Profitability (max +15 points)
        margin = info.get('profitMargins', 0)
        if margin > 0.15:
            score += 15
            factors.append("✅ Strong profit margins (>15%)")
        elif margin > 0.05:
            score += 5
            factors.append("⚠️ Moderate profit margins (5-15%)")
        else:
            score -= 10
            factors.append("❌ Low profit margins (<5%)")
        
        # Leverage (max +10 points)
        dte = info.get('debtToEquity', 0)
        if dte and dte < 50:
            score += 10
            factors.append("✅ Conservative debt (D/E < 50)")
        elif dte > 150:
            score -= 15
            factors.append("❌ High leverage (D/E > 150)")
        else:
            factors.append("⚠️ Moderate debt levels")
        
        # Growth (max +15 points)
        growth = info.get('revenueGrowth', 0)
        if growth and growth > 0.15:
            score += 15
            factors.append("✅ Strong revenue growth (>15%)")
        elif growth and growth < 0:
            score -= 10
            factors.append("❌ Revenue contraction")
        else:
            factors.append("⚠️ Moderate growth")
        
        # Valuation (max +10 points)
        pe = info.get('trailingPE', 0)
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
        
    except Exception as e:
        return json.dumps({'error': str(e)})

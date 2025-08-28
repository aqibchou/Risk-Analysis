#!/usr/bin/env python3
"""
Simple script to analyze risk factors for a single stock
"""

import numpy as np
import pandas as pd
import yfinance as yf
from risk_core import detect_risk_factors

def analyze_stock(ticker: str, period: str = "2y"):
    """
    Analyze a single stock for risk factors
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        period: Time period to analyze ('1y', '2y', '5y', etc.)
    """
    print(f"\n🔍 Analyzing {ticker.upper()} for risk factors...")
    print("=" * 60)
    
    try:
        # Download stock data
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        if data.empty:
            print(f"❌ No data found for {ticker}")
            return
        
        print(f"📊 Downloaded {len(data)} days of data")
        print(f"📅 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        # Calculate returns
        returns = data['Close'].pct_change().dropna()
        print(f"📈 Calculated {len(returns)} daily returns")
        
        # Basic statistics
        print(f"\n📊 Basic Statistics:")
        print(f"   Mean Return: {returns.mean():.4f}")
        print(f"   Std Return: {returns.std():.4f}")
        print(f"   Min Return: {returns.min():.4f}")
        print(f"   Max Return: {returns.max():.4f}")
        
        # Run risk detection
        print(f"\n⚠️  Risk Factor Analysis:")
        risk_results = detect_risk_factors(returns)
        
        print(f"   Entropy: {risk_results['entropy']:.4f}")
        print(f"   Mutual Information: {risk_results['mi_norm']:.4f}")
        print(f"   Hurst Exponent: {risk_results['hurst']:.4f}")
        print(f"   Risk Flag: {risk_results['risk_flag']}")
        
        # Detailed interpretation
        print(f"\n🔍 Risk Interpretation:")
        if risk_results['risk_flag'] == 'structural_time_bomb':
            print("   ⚠️  HIGH RISK: Structural time bomb detected!")
            print("      This indicates:")
            print("      • High persistence (Hurst > 0.5): Price movements tend to continue")
            print("      • High entropy (H_ent > 0.9): Very unpredictable behavior")
            print("      • Low mutual information (MI < 0.05): Little correlation between past and future")
            print("      • Potential for sudden, large price movements")
        elif risk_results['risk_flag'] == 'chaotic_memory':
            print("   🟡 MEDIUM RISK: Chaotic memory detected!")
            print("      This indicates:")
            print("      • High persistence (Hurst > 0.5): Price movements tend to continue")
            print("      • High entropy (H_ent > 0.9): Very unpredictable behavior")
            print("      • Moderate correlation between past and future")
        elif risk_results['risk_flag'] == 'blocked_flow':
            print("   🟡 MEDIUM RISK: Blocked flow detected!")
            print("      This indicates:")
            print("      • Low mutual information (MI < 0.05): Little correlation between past and future")
            print("      • Potential for sudden regime changes")
        else:
            print("   🟢 LOW RISK: Latent risk factors")
            print("      This indicates:")
            print("      • Moderate predictability")
            print("      • Stable correlation patterns")
            print("      • Lower probability of extreme events")
        
        # Additional insights
        print(f"\n💡 Additional Insights:")
        if risk_results['hurst'] > 0.6:
            print("   • High persistence: Current trends likely to continue")
        elif risk_results['hurst'] < 0.4:
            print("   • Low persistence: Mean-reverting behavior likely")
        else:
            print("   • Moderate persistence: Mixed trend/mean-reversion behavior")
            
        if risk_results['entropy'] > 0.8:
            print("   • High entropy: Very unpredictable price movements")
        elif risk_results['entropy'] < 0.5:
            print("   • Low entropy: More predictable price movements")
        else:
            print("   • Moderate entropy: Balanced predictability")
            
        if risk_results['mi_norm'] < 0.1:
            print("   • Low mutual information: Weak correlation between past and future")
        elif risk_results['mi_norm'] > 0.2:
            print("   • High mutual information: Strong correlation between past and future")
        else:
            print("   • Moderate mutual information: Balanced correlation")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"❌ Error analyzing {ticker}: {e}")

def main():
    """Main function"""
    print("🚀 Stock Risk Factor Analyzer")
    print("=" * 60)
    
    while True:
        # Get user input
        ticker = input("\nEnter stock ticker (or 'quit' to exit): ").strip().upper()
        
        if ticker.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not ticker:
            print("❌ Please enter a valid ticker symbol")
            continue
        
        # Get time period
        period = input("Enter time period (1y, 2y, 5y, max) [default: 2y]: ").strip()
        if not period:
            period = "2y"
        
        # Analyze the stock
        analyze_stock(ticker, period)
        
        # Ask if user wants to analyze another stock
        another = input("\nAnalyze another stock? (y/n): ").strip().lower()
        if another not in ['y', 'yes']:
            print("👋 Goodbye!")
            break

if __name__ == "__main__":
    main()

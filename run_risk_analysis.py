import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from risk_core import detect_risk_factors

def download_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Download stock data using yfinance
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        period: Time period to download ('1y', '2y', '5y', etc.)
    
    Returns:
        DataFrame with stock data
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        print(f"Downloaded {len(data)} days of data for {ticker}")
        return data
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None

def calculate_returns(prices: pd.Series) -> np.ndarray:
    """
    Calculate daily returns from price data
    
    Args:
        prices: Series of closing prices
    
    Returns:
        Array of daily returns
    """
    returns = prices.pct_change().dropna()
    return returns.values

def analyze_stock_risk(ticker: str, period: str = "2y") -> dict:
    """
    Analyze risk factors for a given stock
    
    Args:
        ticker: Stock ticker symbol
        period: Time period to analyze
    
    Returns:
        Dictionary with risk analysis results
    """
    print(f"\n=== Risk Analysis for {ticker} ===")
    
    # Download stock data
    data = download_stock_data(ticker, period)
    if data is None:
        return None
    
    # Calculate returns
    returns = calculate_returns(data['Close'])
    print(f"Calculated {len(returns)} daily returns")
    
    # Run risk detection
    risk_results = detect_risk_factors(returns)
    
    # Print results
    print(f"\nRisk Analysis Results:")
    print(f"  Entropy: {risk_results['entropy']:.4f}")
    print(f"  Mutual Information: {risk_results['mi_norm']:.4f}")
    print(f"  Hurst Exponent: {risk_results['hurst']:.4f}")
    print(f"  Risk Flag: {risk_results['risk_flag']}")
    
    # Interpret results
    print(f"\nRisk Interpretation:")
    if risk_results['risk_flag'] == 'structural_time_bomb':
        print("  ⚠️  HIGH RISK: Structural time bomb detected!")
        print("     - High persistence (Hurst > 0.5)")
        print("     - High entropy (H_ent > 0.9)")
        print("     - Low mutual information (MI < 0.05)")
    elif risk_results['risk_flag'] == 'chaotic_memory':
        print("  🟡 MEDIUM RISK: Chaotic memory detected!")
        print("     - High persistence (Hurst > 0.5)")
        print("     - High entropy (H_ent > 0.9)")
    elif risk_results['risk_flag'] == 'blocked_flow':
        print("  🟡 MEDIUM RISK: Blocked flow detected!")
        print("     - Low mutual information (MI < 0.05)")
    else:
        print("  🟢 LOW RISK: Latent risk factors")
    
    return risk_results

def plot_risk_analysis(ticker: str, risk_results: dict, data: pd.DataFrame):
    """
    Create visualization of the risk analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Risk Analysis for {ticker}', fontsize=16)
    
    # Plot 1: Price and Returns
    ax1 = axes[0, 0]
    ax1.plot(data.index, data['Close'], label='Close Price', color='blue')
    ax1.set_title('Stock Price')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True)
    
    ax1_twin = ax1.twinx()
    returns = data['Close'].pct_change().dropna()
    ax1_twin.plot(returns.index, returns, label='Returns', color='red', alpha=0.7)
    ax1_twin.set_ylabel('Returns', color='red')
    ax1_twin.legend(loc='upper right')
    
    # Plot 2: Risk Metrics
    ax2 = axes[0, 1]
    metrics = ['Entropy', 'Mutual Info', 'Hurst']
    values = [risk_results['entropy'], risk_results['mi_norm'], risk_results['hurst']]
    colors = ['green' if v < 0.5 else 'orange' if v < 0.8 else 'red' for v in values]
    
    bars = ax2.bar(metrics, values, color=colors, alpha=0.7)
    ax2.set_title('Risk Metrics')
    ax2.set_ylabel('Value')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 3: Returns Distribution
    ax3 = axes[1, 0]
    returns = data['Close'].pct_change().dropna()
    ax3.hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_title('Returns Distribution')
    ax3.set_xlabel('Returns')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Risk Flag
    ax4 = axes[1, 1]
    risk_levels = {
        'latent': 1,
        'blocked_flow': 2,
        'chaotic_memory': 3,
        'structural_time_bomb': 4
    }
    
    risk_level = risk_levels.get(risk_results['risk_flag'], 0)
    risk_colors = ['green', 'orange', 'orange', 'red']
    risk_labels = ['Low', 'Medium', 'Medium', 'High']
    
    ax4.bar(['Risk Level'], [risk_level], color=risk_colors[risk_level-1], alpha=0.7)
    ax4.set_title('Risk Level Assessment')
    ax4.set_ylabel('Risk Level (1=Low, 4=High)')
    ax4.set_ylim(0, 5)
    ax4.text(0, risk_level + 0.1, risk_labels[risk_level-1], 
             ha='center', va='bottom', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_risk_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to run risk analysis on stocks
    """
    # List of stocks to analyze
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    print("Stock Risk Factor Analysis")
    print("=" * 50)
    
    all_results = {}
    
    for stock in stocks:
        try:
            results = analyze_stock_risk(stock)
            if results:
                all_results[stock] = results
                
                # Download data for plotting
                data = download_stock_data(stock, "2y")
                if data is not None:
                    plot_risk_analysis(stock, results, data)
                
                print("\n" + "="*50)
        except Exception as e:
            print(f"Error analyzing {stock}: {e}")
            continue
    
    # Summary comparison
    if all_results:
        print("\n=== Summary Comparison ===")
        summary_df = pd.DataFrame(all_results).T
        print(summary_df[['entropy', 'mi_norm', 'hurst', 'risk_flag']])
        
        # Find highest risk stocks
        high_risk = [stock for stock, results in all_results.items() 
                     if results['risk_flag'] in ['structural_time_bomb', 'chaotic_memory']]
        
        if high_risk:
            print(f"\n⚠️  High Risk Stocks: {', '.join(high_risk)}")
        else:
            print("\n🟢 All analyzed stocks show low risk levels")

if __name__ == "__main__":
    main()

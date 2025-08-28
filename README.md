# Stock Risk Factor Analysis

This project provides tools that use concepts from physics and mathematics to detect risk factors in stocks using advanced statistical methods including entropy analysis, mutual information, and Hurst exponent calculations. This project provides real risk measurements incomporable to ignorant "risk=volatility" models.

## Features

- **Entropy Analysis**: Measures the unpredictability of stock returns
- **Mutual Information**: Quantifies the correlation between past and future returns
- **Hurst Exponent**: Determines if price movements are persistent or mean-reverting
- **Risk Classification**: Automatically categorizes stocks into risk levels

## Risk Categories

### 🟢 Low Risk (Latent)
- Moderate predictability
- Stable correlation patterns
- Lower probability of extreme events

### 🟡 Medium Risk
- **Chaotic Memory**: High persistence + high entropy
- **Blocked Flow**: Low mutual information (weak correlation)

### ⚠️ High Risk (Structural Time Bomb)
- High persistence (Hurst > 0.5)
- High entropy (H_ent > 0.9)
- Low mutual information (MI < 0.05)


## Installation

1. Install required dependencies:
```bash
pip3 install -r requirements.txt
```

## Usage

### Batch Analysis
Run risk analysis on multiple stocks at once:

```bash
python3 run_risk_analysis.py
```

This will analyze: AAPL, MSFT, GOOGL, TSLA, NVDA

### Single Stock Analysis
Analyze a specific stock interactively:

```bash
python3 analyze_single_stock.py
```

Then enter the stock ticker when prompted.

### Custom Analysis
Use the functions directly in your code:

```python
from risk_core import detect_risk_factors
import yfinance as yf

# Download stock data
stock = yf.Ticker('AAPL')
data = stock.history(period='2y')
returns = data['Close'].pct_change().dropna()

# Analyze risk factors
risk_results = detect_risk_factors(returns)
print(risk_results)
```

## Understanding the Results

### Entropy (0-1)
- **Low (< 0.5)**: More predictable price movements
- **Medium (0.5-0.8)**: Balanced predictability
- **High (> 0.8)**: Very unpredictable behavior

### Mutual Information (0-1)
- **Low (< 0.1)**: Weak correlation between past and future
- **Medium (0.1-0.2)**: Balanced correlation
- **High (> 0.2)**: Strong correlation between past and future

### Hurst Exponent
- **< 0.4**: Mean-reverting behavior (prices tend to return to average)
- **0.4-0.6**: Mixed behavior
- **> 0.6**: Persistent behavior (trends tend to continue)

## Example Output

```
🔍 Analyzing AAPL for risk factors...
============================================================
📊 Downloaded 502 days of data
📅 Date range: 2023-08-28 to 2025-08-28
📈 Calculated 501 daily returns

📊 Basic Statistics:
   Mean Return: 0.0008
   Std Return: 0.0189
   Min Return: -0.0523
   Max Return: 0.0789

⚠️  Risk Factor Analysis:
   Entropy: 0.5896
   Mutual Information: 0.1023
   Hurst Exponent: 0.5546
   Risk Flag: latent

🔍 Risk Interpretation:
   🟢 LOW RISK: Latent risk factors
   This indicates:
   • Moderate predictability
   • Stable correlation patterns
   • Lower probability of extreme events
```

## Files

- `risk_core.py`: Core risk detection functions
- `run_risk_analysis.py`: Batch analysis of multiple stocks
- `analyze_single_stock.py`: Interactive single stock analysis
- `requirements.txt`: Python dependencies
- `README.md`: This documentation

## Technical Details

The risk detection algorithm combines three key metrics:

1. **Shannon Entropy**: Measures information content and unpredictability
2. **Normalized Mutual Information**: Quantifies temporal dependencies
3. **Hurst Exponent (R/S Analysis)**: Determines persistence vs. mean-reversion

The algorithm automatically flags stocks based on threshold combinations of these metrics, providing early warning of potential risk factors.



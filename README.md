# AI Crypto Trading Assistant - Binance Edition

A sophisticated AI-powered cryptocurrency trading assistant that combines real-time market data from Binance with advanced AI analysis using Groq's language models. This application provides comprehensive trading signals, technical analysis, and market insights to help traders make informed decisions.

## Features

### Core Capabilities
- **Real-time Market Data**: Live price feeds from Binance API
- **AI-Powered Analysis**: Advanced trading signal generation using Groq's LLM
- **Technical Indicators**: RSI, SMA, Bollinger Bands, and volume analysis
- **Anomaly Detection**: Machine learning-based price anomaly detection
- **Multi-symbol Support**: Support for major cryptocurrencies with intelligent symbol mapping
- **RESTful API**: Clean, documented API endpoints for easy integration

### AI Analysis Features
- **Trading Signals**: BUY/SELL/HOLD recommendations with confidence scores
- **Market Sentiment**: Automated bullish/bearish/neutral sentiment analysis
- **Risk Assessment**: Comprehensive risk analysis for each trading signal
- **Portfolio Insights**: AI-powered portfolio analysis and rebalancing suggestions

### Technical Analysis
- **Moving Averages**: SMA-20, SMA-50
- **RSI**: Relative Strength Index for momentum analysis
- **Bollinger Bands**: Volatility and price action analysis
- **Volume Analysis**: Volume ratio and trend analysis
- **Price Anomalies**: ML-based detection of unusual price movements

## Technology Stack

- **Backend**: FastAPI (Python)
- **AI/ML**: Groq API, LangChain, scikit-learn
- **Data Source**: Binance API
- **Technical Analysis**: NumPy, Pandas
- **API Framework**: FastAPI with CORS support
- **Deployment**: Uvicorn ASGI server

## Installation

### Prerequisites
- Python 3.8+
- Groq API key
- Internet connection for Binance API access

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/bellonbits/CryptoAI.git
   cd CryptoAI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys**
   - Update the `GROQ_API_KEY` in the code with your Groq API key
   - No Binance API key required for public market data

4. **Run the application**
   ```bash
   python main.py
   ```

The server will start on `http://localhost:8000`

### Dependencies
```
fastapi
uvicorn
requests
pandas
numpy
scikit-learn
joblib
langchain-groq
langchain
```

## API Endpoints

### Core Endpoints

#### Get AI Trading Signal
```
GET /ai-signal/{symbol}
```
Returns comprehensive AI analysis including:
- ML-based trading prediction
- Technical indicators
- Market sentiment
- Anomaly detection
- Groq AI analysis

**Example**: `GET /ai-signal/btc`

#### Get Market Data
```
GET /market/{symbol}
```
Returns real-time market data from Binance:
- Current price
- 24h price change
- Volume data
- Order book information

#### Get Historical Data
```
GET /historical/{symbol}?interval=1h&limit=100
```
Returns historical candlestick data with configurable intervals.

#### Get Technical Analysis
```
GET /technical/{symbol}
```
Returns technical indicators and anomaly detection results.

### Utility Endpoints

#### Supported Symbols
```
GET /supported-symbols
```
Returns list of supported cryptocurrency symbols.

#### Health Check
```
GET /health
```
Returns system health status.

## Supported Cryptocurrencies

The system includes intelligent symbol mapping for major cryptocurrencies:

| Symbol | Binance Pair | Common Names |
|--------|-------------|--------------|
| BTC | BTCUSDT | Bitcoin |
| ETH | ETHUSDT | Ethereum |
| BNB | BNBUSDT | Binance Coin |
| ADA | ADAUSDT | Cardano |
| DOT | DOTUSDT | Polkadot |
| XRP | XRPUSDT | Ripple |
| SOL | SOLUSDT | Solana |
| DOGE | DOGEUSDT | Dogecoin |
| MATIC | MATICUSDT | Polygon |
| AVAX | AVAXUSDT | Avalanche |

*And many more...*

## AI Analysis Example

```json
{
  "symbol": "btc",
  "binance_symbol": "BTCUSDT",
  "current_data": {
    "current_price": 43250.50,
    "price_change_percent_24h": 2.45,
    "volume_24h": 25430000000
  },
  "ml_prediction": {
    "signal": "BUY",
    "confidence": 78,
    "buy_signals": 4,
    "sell_signals": 1
  },
  "technical_indicators": {
    "rsi": 65.2,
    "sma_20": 42800.30,
    "bb_upper": 45000.00,
    "bb_lower": 40000.00
  },
  "market_sentiment": "bullish",
  "groq_analysis": "Based on technical analysis, BTC shows strong bullish momentum..."
}
```

## Security & Best Practices

### Security Features
- **No Private Keys**: Uses only public Binance API endpoints
- **CORS Protection**: Configurable CORS middleware
- **Error Handling**: Comprehensive error handling and logging
- **Rate Limiting**: Respects Binance API rate limits

### Best Practices
- **Paper Trading**: Always test strategies with paper trading first
- **Risk Management**: Never invest more than you can afford to lose
- **Diversification**: Use insights for portfolio diversification
- **Market Research**: Combine AI insights with fundamental analysis

## Usage Examples

### Basic Trading Signal
```python
import requests

# Get AI trading signal for Bitcoin
response = requests.get("http://localhost:8000/ai-signal/btc")
signal_data = response.json()

print(f"Signal: {signal_data['ml_prediction']['signal']}")
print(f"Confidence: {signal_data['ml_prediction']['confidence']}%")
```

### Market Data Monitoring
```python
# Monitor multiple cryptocurrencies
symbols = ['btc', 'eth', 'ada', 'sol']

for symbol in symbols:
    response = requests.get(f"http://localhost:8000/market/{symbol}")
    data = response.json()
    print(f"{symbol.upper()}: ${data['current_price']:.2f} ({data['price_change_percent_24h']:.2f}%)")
```

## Disclaimer

**IMPORTANT**: This software is for educational and informational purposes only. 

- **Not Financial Advice**: This tool does not provide financial advice
- **Trading Risks**: Cryptocurrency trading involves substantial risk
- **No Guarantees**: Past performance does not guarantee future results
- **Use at Own Risk**: Users are responsible for their trading decisions

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- New technical indicators
- Additional AI analysis features
- Bug fixes and improvements
- Documentation updates

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Open an issue on GitHub
- Check the API documentation at `http://localhost:8000/docs`
- Review the health check endpoint for system status

## Version History

### Version 2.0
- Added Groq AI integration
- Enhanced technical analysis
- Improved anomaly detection
- RESTful API design
- Comprehensive error handling

---

**Happy Trading!**

*Remember: The best trading strategy combines AI insights with human judgment and proper risk management.*

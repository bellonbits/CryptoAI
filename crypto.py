import os
import json
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Groq and LangChain imports
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


# Initialize FastAPI
app = FastAPI(title="AI Crypto Trading Assistant", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
GROQ_API_KEY = ""
BINANCE_BASE_URL = "https://api.binance.com/api/v3"

# Symbol mapping for common cryptocurrencies
SYMBOL_MAPPING = {
    'btc': 'BTCUSDT',
    'bitcoin': 'BTCUSDT',
    'eth': 'ETHUSDT',
    'ethereum': 'ETHUSDT',
    'bnb': 'BNBUSDT',
    'ada': 'ADAUSDT',
    'cardano': 'ADAUSDT',
    'dot': 'DOTUSDT',
    'polkadot': 'DOTUSDT',
    'xrp': 'XRPUSDT',
    'ripple': 'XRPUSDT',
    'ltc': 'LTCUSDT',
    'litecoin': 'LTCUSDT',
    'link': 'LINKUSDT',
    'chainlink': 'LINKUSDT',
    'sol': 'SOLUSDT',
    'solana': 'SOLUSDT',
    'doge': 'DOGEUSDT',
    'dogecoin': 'DOGEUSDT',
    'matic': 'MATICUSDT',
    'polygon': 'MATICUSDT',
    'avax': 'AVAXUSDT',
    'avalanche': 'AVAXUSDT',
    'shib': 'SHIBUSDT',
    'uni': 'UNIUSDT',
    'uniswap': 'UNIUSDT'
}

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

class GroqAIAssistant:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama3-70b-8192",  # Updated to current model
            temperature=0.3
        )
        self.setup_chains()

    def setup_chains(self):
        """Setup specialized LLM chains"""
        # Trading signal analysis chain
        signal_template = """Analyze this market data for {symbol}:
        Price Data: {prices}
        Technical Indicators: {indicators}
        Market Sentiment: {sentiment}

        Based on this data, provide:
        1. Trading recommendation (BUY/SELL/HOLD)
        2. Confidence level (1-100)
        3. Risk assessment
        4. Key factors influencing the decision
        5. Suggested entry/exit points

        Keep the response concise and actionable."""

        self.signal_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(signal_template)
        )

        # Portfolio analysis chain
        portfolio_template = """Analyze this portfolio:
        Holdings: {holdings}
        Market Conditions: {market_data}

        Provide:
        1. Portfolio health assessment
        2. Rebalancing recommendations
        3. Risk analysis
        4. Diversification suggestions

        Keep recommendations practical and specific."""

        self.portfolio_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(portfolio_template)
        )

class BinanceMarketData:
    def __init__(self):
        self.base_url = BINANCE_BASE_URL
        self.anomaly_detector = IsolationForest(contamination=0.05)

    def normalize_symbol(self, symbol: str) -> str:
        """Convert symbol to Binance format"""
        symbol = symbol.lower().strip()

        # Check if it's in our mapping
        if symbol in SYMBOL_MAPPING:
            return SYMBOL_MAPPING[symbol]

        # If it's already in correct format
        if symbol.upper().endswith('USDT'):
            return symbol.upper()

        # Try to append USDT
        return symbol.upper() + 'USDT'

    async def get_price_data(self, symbol: str) -> Dict:
        """Get comprehensive price data from Binance"""
        binance_symbol = self.normalize_symbol(symbol)

        try:
            # Get 24hr ticker statistics
            ticker_url = f"{self.base_url}/ticker/24hr"
            ticker_resp = requests.get(ticker_url, params={'symbol': binance_symbol})

            if ticker_resp.status_code != 200:
                raise Exception(f"Binance API error: {ticker_resp.status_code}")

            ticker_data = ticker_resp.json()

            # Get current price
            price_url = f"{self.base_url}/ticker/price"
            price_resp = requests.get(price_url, params={'symbol': binance_symbol})
            price_data = price_resp.json()

            # Get order book depth
            depth_url = f"{self.base_url}/depth"
            depth_resp = requests.get(depth_url, params={'symbol': binance_symbol, 'limit': 10})
            depth_data = depth_resp.json()

            # Get recent trades
            trades_url = f"{self.base_url}/trades"
            trades_resp = requests.get(trades_url, params={'symbol': binance_symbol, 'limit': 100})
            trades_data = trades_resp.json()

            return {
                'symbol': binance_symbol,
                'current_price': float(price_data['price']),
                'price_change_24h': float(ticker_data['priceChange']),
                'price_change_percent_24h': float(ticker_data['priceChangePercent']),
                'high_24h': float(ticker_data['highPrice']),
                'low_24h': float(ticker_data['lowPrice']),
                'volume_24h': float(ticker_data['volume']),
                'quote_volume_24h': float(ticker_data['quoteVolume']),
                'open_price': float(ticker_data['openPrice']),
                'bid_price': float(depth_data['bids'][0][0]) if depth_data['bids'] else 0,
                'ask_price': float(depth_data['asks'][0][0]) if depth_data['asks'] else 0,
                'bid_qty': float(depth_data['bids'][0][1]) if depth_data['bids'] else 0,
                'ask_qty': float(depth_data['asks'][0][1]) if depth_data['asks'] else 0,
                'trade_count': int(ticker_data['count']),
                'last_trade_time': int(trades_data[0]['time']) if trades_data else 0,
                'weighted_avg_price': float(ticker_data['weightedAvgPrice'])
            }

        except Exception as e:
            print(f"Error fetching Binance data for {binance_symbol}: {e}")
            return {
                'symbol': binance_symbol,
                'error': str(e),
                'current_price': 0
            }

    async def get_historical_data(self, symbol: str, interval: str = '1h', limit: int = 100) -> List[Dict]:
        """Get historical kline/candlestick data"""
        binance_symbol = self.normalize_symbol(symbol)

        try:
            klines_url = f"{self.base_url}/klines"
            params = {
                'symbol': binance_symbol,
                'interval': interval,
                'limit': limit
            }

            resp = requests.get(klines_url, params=params)
            if resp.status_code != 200:
                return []

            klines = resp.json()
            historical_data = []

            for kline in klines:
                historical_data.append({
                    'timestamp': int(kline[0]),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'close_time': int(kline[6]),
                    'quote_volume': float(kline[7]),
                    'trade_count': int(kline[8])
                })

            return historical_data

        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return []

    def calculate_technical_indicators(self, historical_data: List[Dict]) -> Dict:
        """Calculate basic technical indicators"""
        if len(historical_data) < 20:
            return {'error': 'Insufficient data for technical indicators'}

        closes = [float(data['close']) for data in historical_data]
        highs = [float(data['high']) for data in historical_data]
        lows = [float(data['low']) for data in historical_data]
        volumes = [float(data['volume']) for data in historical_data]

        # Simple Moving Averages
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else np.mean(closes)

        # RSI calculation (simplified)
        price_changes = np.diff(closes)
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))

        # Bollinger Bands
        bb_period = min(20, len(closes))
        bb_sma = np.mean(closes[-bb_period:])
        bb_std = np.std(closes[-bb_period:])
        bb_upper = bb_sma + (2 * bb_std)
        bb_lower = bb_sma - (2 * bb_std)

        # Volume analysis
        vol_sma = np.mean(volumes[-20:])
        vol_ratio = volumes[-1] / vol_sma if vol_sma > 0 else 1

        return {
            'sma_20': float(sma_20),
            'sma_50': float(sma_50),
            'rsi': float(rsi),
            'bb_upper': float(bb_upper),
            'bb_middle': float(bb_sma),
            'bb_lower': float(bb_lower),
            'volume_ratio': float(vol_ratio),
            'price_position': 'above_sma20' if closes[-1] > sma_20 else 'below_sma20'
        }

    def detect_anomalies(self, historical_data: List[Dict]) -> List[Dict]:
        """Detect price anomalies"""
        if len(historical_data) < 10:
            return []

        try:
            # Extract features for anomaly detection
            features = []
            for data in historical_data:
                features.append([
                    data['close'],
                    data['volume'],
                    data['high'] - data['low'],  # Range
                    (data['close'] - data['open']) / data['open'] * 100  # Percent change
                ])

            features = np.array(features)

            # Fit and predict anomalies
            self.anomaly_detector.fit(features)
            predictions = self.anomaly_detector.predict(features)
            scores = self.anomaly_detector.decision_function(features)

            anomalies = []
            for i, (pred, score) in enumerate(zip(predictions, scores)):
                if pred == -1:  # Anomaly detected
                    anomalies.append({
                        'timestamp': historical_data[i]['timestamp'],
                        'price': historical_data[i]['close'],
                        'severity': 'high' if score < -0.5 else 'medium',
                        'anomaly_score': float(score)
                    })

            return anomalies[-5:]  # Return last 5 anomalies

        except Exception as e:
            print(f"Anomaly detection error: {e}")
            return []

class TradingSignalGenerator:
    def __init__(self, groq_assistant: GroqAIAssistant, market_data: BinanceMarketData):
        self.groq = groq_assistant
        self.market_data = market_data

    def _ml_predict(self, symbol: str, current_data: Dict, historical_data: List[Dict],
                   technical_indicators: Dict) -> Dict:
        """Generate ML-based trading prediction"""
        try:
            if 'error' in current_data or not historical_data:
                return {'signal': 'HOLD', 'confidence': 0, 'reason': 'Insufficient data'}

            # Extract features
            current_price = current_data['current_price']
            price_change_24h = current_data['price_change_percent_24h']
            volume_24h = current_data['volume_24h']

            # Technical indicator features
            rsi = technical_indicators.get('rsi', 50)
            sma_20 = technical_indicators.get('sma_20', current_price)
            vol_ratio = technical_indicators.get('volume_ratio', 1)

            # Simple rule-based prediction
            buy_signals = 0
            sell_signals = 0

            # Price trend analysis
            if price_change_24h > 5:
                buy_signals += 2
            elif price_change_24h < -5:
                sell_signals += 2

            # RSI analysis
            if rsi < 30:
                buy_signals += 2  # Oversold
            elif rsi > 70:
                sell_signals += 2  # Overbought

            # Price vs SMA
            if current_price > sma_20:
                buy_signals += 1
            else:
                sell_signals += 1

            # Volume analysis
            if vol_ratio > 1.5:
                if price_change_24h > 0:
                    buy_signals += 1
                else:
                    sell_signals += 1

            # Determine signal
            if buy_signals > sell_signals + 1:
                signal = 'BUY'
                confidence = min(buy_signals * 15, 95)
            elif sell_signals > buy_signals + 1:
                signal = 'SELL'
                confidence = min(sell_signals * 15, 95)
            else:
                signal = 'HOLD'
                confidence = 50

            return {
                'signal': signal,
                'confidence': float(confidence),
                'buy_signals': int(buy_signals),
                'sell_signals': int(sell_signals),
                'features': {
                    'price_change_24h': float(price_change_24h),
                    'rsi': float(rsi),
                    'volume_ratio': float(vol_ratio),
                    'price_vs_sma20': float((current_price - sma_20) / sma_20 * 100)
                }
            }

        except Exception as e:
            print(f"ML prediction error: {e}")
            return {'signal': 'HOLD', 'confidence': 0, 'reason': str(e)}

    def _get_market_sentiment(self, current_data: Dict, technical_indicators: Dict) -> str:
        """Determine market sentiment"""
        if 'error' in current_data:
            return 'neutral'

        price_change = current_data.get('price_change_percent_24h', 0)
        rsi = technical_indicators.get('rsi', 50)

        if price_change > 3 and rsi < 70:
            return 'bullish'
        elif price_change < -3 and rsi > 30:
            return 'bearish'
        else:
            return 'neutral'

    async def generate_signal(self, symbol: str) -> Dict:
        """Generate comprehensive AI trading signal"""
        try:
            # Get current market data
            current_data = await self.market_data.get_price_data(symbol)

            # Get historical data
            historical_data = await self.market_data.get_historical_data(symbol)

            # Calculate technical indicators
            technical_indicators = self.market_data.calculate_technical_indicators(historical_data)

            # Detect anomalies
            anomalies = self.market_data.detect_anomalies(historical_data)

            # Generate ML prediction
            ml_signal = self._ml_predict(symbol, current_data, historical_data, technical_indicators)

            # Get market sentiment
            sentiment = self._get_market_sentiment(current_data, technical_indicators)

            # Get Groq AI analysis
            groq_analysis = "Analysis unavailable"
            try:
                groq_response = await self.groq.signal_chain.ainvoke({
                    'symbol': symbol,
                    'prices': json.dumps(convert_numpy_types(current_data)),
                    'indicators': json.dumps(convert_numpy_types(technical_indicators)),
                    'sentiment': sentiment
                })
                groq_analysis = groq_response.get('text', str(groq_response))
            except Exception as e:
                print(f"Groq analysis error: {e}")

            return convert_numpy_types({
                'symbol': symbol,
                'binance_symbol': current_data.get('symbol', symbol),
                'current_data': current_data,
                'ml_prediction': ml_signal,
                'technical_indicators': technical_indicators,
                'market_sentiment': sentiment,
                'anomalies': anomalies,
                'groq_analysis': groq_analysis,
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            print(f"Signal generation error: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Initialize services
groq_assistant = GroqAIAssistant()
market_data = BinanceMarketData()
signal_generator = TradingSignalGenerator(groq_assistant, market_data)

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Crypto Trading Assistant - Binance Edition",
        "version": "2.0",
        "data_source": "Binance API",
        "endpoints": [
            "/ai-signal/{symbol}",
            "/market/{symbol}",
            "/historical/{symbol}",
            "/supported-symbols"
        ]
    }

@app.get("/supported-symbols")
async def get_supported_symbols():
    """Get list of supported symbols"""
    return {
        "supported_symbols": list(SYMBOL_MAPPING.keys()),
        "note": "You can also use any symbol that exists on Binance (e.g., BTCUSDT, ETHUSDT)"
    }

@app.get("/ai-signal/{symbol}")
async def get_ai_signal(symbol: str):
    """Get comprehensive AI trading signal"""
    try:
        return await signal_generator.generate_signal(symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/{symbol}")
async def get_market_data(symbol: str):
    """Get real-time market data"""
    try:
        data = await market_data.get_price_data(symbol)
        return convert_numpy_types(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/historical/{symbol}")
async def get_historical_data(symbol: str, interval: str = "1h", limit: int = 100):
    """Get historical price data"""
    try:
        data = await market_data.get_historical_data(symbol, interval, limit)
        return convert_numpy_types({
            'symbol': symbol,
            'interval': interval,
            'data': data,
            'count': len(data)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/technical/{symbol}")
async def get_technical_analysis(symbol: str):
    """Get technical analysis for a symbol"""
    try:
        historical_data = await market_data.get_historical_data(symbol)
        technical_indicators = market_data.calculate_technical_indicators(historical_data)
        anomalies = market_data.detect_anomalies(historical_data)

        return convert_numpy_types({
            'symbol': symbol,
            'technical_indicators': technical_indicators,
            'anomalies': anomalies,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "binance_api": "active",
            "groq_ai": "active",
            "signal_generator": "active"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

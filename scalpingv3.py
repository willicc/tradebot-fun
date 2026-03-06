import ccxt
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import warnings
from datetime import datetime
import json

warnings.filterwarnings("ignore")

# =============================================================================
# KONFIGURASI API & EXCHANGE
# =============================================================================
BINANCE_API_KEY = 'GANTI_DENGAN_API_KEY_ANDA'
BINANCE_SECRET_KEY = 'GANTI_DENGAN_SECRET_KEY_ANDA'

exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET_KEY,
    'options': {
        'defaultType': 'future',
        'enableRateLimit': True,
    },
})

# =============================================================================
# KONFIGURASI STRATEGI 1-MINUTE
# =============================================================================
TIMEFRAME_PRIMARY = '1m'      # Primary untuk entry
TIMEFRAME_CONFIRM = '5m'      # Confirmation
TIMEFRAME_TREND = '15m'       # Trend direction

# Indikator untuk 1-menit (Optimized)
EMA_FAST = 5
EMA_MID = 12
EMA_SLOW = 26
RSI_PERIOD = 7                # Faster untuk 1m
RSI_OVERBOUGHT = 75           # Lebih ketat
RSI_OVERSOLD = 25             # Lebih ketat
STOCH_K_PERIOD = 5
STOCH_RSI_PERIOD = 5

# Trading Configuration
POSITION_SIZE_USDT = 100      # Kecil untuk scalping 1m
LEVERAGE = 3                  # Conservative
ATR_SL_MULTIPLIER = 0.7       # Tight stops
ATR_TP_MULTIPLIER = 1.2       # Target 0.8-1.2% profit

# Bot Settings
SCAN_INTERVAL = 60            # 1 menit
MAX_POSITIONS = 3
MIN_VOLUME_USDT = 500_000     # Min volume filter

# ML Settings
ML_TRAINING_CANDLES = 150     # Training period
ML_CONFIDENCE_THRESHOLD = 0.65  # Stricter threshold

# Risk Management
MAX_DAILY_LOSS_USDT = 2000
PROFIT_TAKING_LEVEL = 1000

# =============================================================================
# COLOR LOGGING
# =============================================================================
class Logger:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    @staticmethod
    def info(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {Logger.OKCYAN}[INFO]{Logger.ENDC} {msg}")
    
    @staticmethod
    def success(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {Logger.OKGREEN}[✓]{Logger.ENDC} {msg}")
    
    @staticmethod
    def warning(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {Logger.WARNING}[!]{Logger.ENDC} {msg}")
    
    @staticmethod
    def error(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {Logger.FAIL}[✗]{Logger.ENDC} {msg}")
    
    @staticmethod
    def signal(direction, symbol, confidence, entry_price):
        if direction == "LONG":
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {Logger.OKGREEN}🟢 LONG SIGNAL{Logger.ENDC} {symbol} | Confidence: {confidence:.1%} | Price: ${entry_price:.8f}\n")
        else:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {Logger.FAIL}🔴 SHORT SIGNAL{Logger.ENDC} {symbol} | Confidence: {confidence:.1%} | Price: ${entry_price:.8f}\n")

# =============================================================================
# FUNGSI 1: CALCULATE FEATURES (1-MINUTE OPTIMIZED)
# =============================================================================
def calculate_indicators_1m(df):
    """Hitung semua indikator optimal untuk 1-menit"""
    df = df.copy()
    
    # EMAs
    df['EMA_5'] = ta.ema(df['close'], length=EMA_FAST)
    df['EMA_12'] = ta.ema(df['close'], length=EMA_MID)
    df['EMA_26'] = ta.ema(df['close'], length=EMA_SLOW)
    
    # RSI (Faster untuk 1m)
    df['RSI_7'] = ta.rsi(df['close'], length=RSI_PERIOD)
    
    # MACD (Shorter period untuk 1m)
    macd_result = ta.macd(df['close'], fast=5, slow=13, signal=5)
    df['MACD'] = macd_result.iloc[:, 0]
    df['MACD_SIGNAL'] = macd_result.iloc[:, 1]
    df['MACD_HIST'] = macd_result.iloc[:, 2]
    
    # Stochastic RSI (Faster)
    stoch_rsi = ta.stochrsi(df['close'], length=STOCH_RSI_PERIOD, rsi_length=7, k=3, d=3)
    df['STOCH_K'] = stoch_rsi.iloc[:, 0]
    df['STOCH_D'] = stoch_rsi.iloc[:, 1]
    
    # Bollinger Bands (Tighter untuk 1m)
    bb = ta.bbands(df['close'], length=15, std=1.5)
    df['BB_UPPER'] = bb.iloc[:, 2]
    df['BB_MID'] = bb.iloc[:, 1]
    df['BB_LOWER'] = bb.iloc[:, 0]
    
    # ATR (untuk SL/TP)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=10)
    
    # ADX (Trend Strength)
    adx_result = ta.adx(df['high'], df['low'], df['close'], length=10)
    df['ADX'] = adx_result.iloc[:, 0]
    
    # CCI (Commodity Channel Index)
    df['CCI'] = ta.cci(df['high'], df['low'], df['close'], length=10)
    
    # Volume
    df['VOLUME_SMA'] = df['volume'].rolling(20).mean()
    df['VOLUME_RATIO'] = df['volume'] / df['VOLUME_SMA']
    
    # Price Action
    df['BODY'] = abs(df['close'] - df['open'])
    df['RANGE'] = df['high'] - df['low']
    df['UPPER_WICK'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['LOWER_WICK'] = df[['open', 'close']].min(axis=1) - df['low']
    
    # Momentum
    df['MOMENTUM'] = df['close'].pct_change(3)
    df['ROC_5'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)) * 100
    
    # EMA Alignment
    df['EMA_5_12_RATIO'] = ((df['EMA_5'] - df['EMA_12']) / df['EMA_12']) * 100
    df['EMA_12_26_RATIO'] = ((df['EMA_12'] - df['EMA_26']) / df['EMA_26']) * 100
    
    # Volatility
    df['VOLATILITY'] = df['close'].pct_change().rolling(10).std() * 100
    
    # Price Position in Range
    df['HIGHEST'] = df['high'].rolling(20).max()
    df['LOWEST'] = df['low'].rolling(20).min()
    df['PRICE_POSITION'] = ((df['close'] - df['LOWEST']) / (df['HIGHEST'] - df['LOWEST'])) * 100
    
    return df.fillna(0)

# =============================================================================
# FUNGSI 2: TRAIN ML MODEL
# =============================================================================
def train_ml_model(df):
    """Train model untuk prediksi arah"""
    try:
        df = df.dropna()
        if len(df) < ML_TRAINING_CANDLES:
            return None
        
        # Feature selection
        features = [
            'EMA_5', 'EMA_12', 'EMA_26', 'RSI_7',
            'MACD', 'MACD_SIGNAL', 'MACD_HIST',
            'BB_UPPER', 'BB_MID', 'BB_LOWER',
            'STOCH_K', 'STOCH_D', 'CCI',
            'ATR', 'ADX', 'VOLUME_RATIO',
            'MOMENTUM', 'ROC_5',
            'EMA_5_12_RATIO', 'EMA_12_26_RATIO',
            'VOLATILITY', 'PRICE_POSITION'
        ]
        
        # Target: 1 = UP 0.5%+, -1 = DOWN 0.5%-, 0 = Neutral
        df['TARGET'] = df['close'].shift(-5).pct_change().apply(
            lambda x: 1 if x > 0.005 else (-1 if x < -0.005 else 0)
        )
        
        df_ml = df[features + ['TARGET']].dropna()
        if len(df_ml) < ML_TRAINING_CANDLES:
            return None
        
        # Split train
        train_data = df_ml.tail(ML_TRAINING_CANDLES)
        X_train = train_data[features].values
        y_train = train_data['TARGET'].values
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        return {'model': model, 'scaler': scaler, 'features': features}
    
    except Exception as e:
        Logger.error(f"ML Training error: {e}")
        return None

# =============================================================================
# FUNGSI 3: PREDICT DIRECTION
# =============================================================================
def predict_direction(df, ml_model):
    """Predict naik/turun dengan ML"""
    try:
        if ml_model is None:
            return None, 0
        
        df = df.dropna()
        features = ml_model['features']
        current_features = df[features].iloc[-1:].values
        
        scaler = ml_model['scaler']
        model = ml_model['model']
        
        current_scaled = scaler.transform(current_features)
        prediction = model.predict(current_scaled)[0]
        proba = model.predict_proba(current_scaled).max()
        
        return prediction, proba
    
    except Exception as e:
        Logger.error(f"Prediction error: {e}")
        return None, 0

# =============================================================================
# FUNGSI 4: DETECT LONG SIGNAL (7 Filters)
# =============================================================================
def detect_long_signal(df, ml_pred, ml_proba):
    """Detect LONG dengan 7 filter untuk 70%+ winrate"""
    if len(df) < 2:
        return False, 0
    
    c = df.iloc[-1]  # Current
    p = df.iloc[-2]  # Previous
    
    filters_met = 0
    
    # Filter 1: EMA Alignment (5 > 12 > 26)
    if c['EMA_5'] > c['EMA_12'] > c['EMA_26']:
        filters_met += 1
    
    # Filter 2: EMA Crossover (5 crossing above 12)
    if p['EMA_5'] <= p['EMA_12'] and c['EMA_5'] > c['EMA_12']:
        filters_met += 1
    
    # Filter 3: MACD Bullish (MACD > Signal)
    if c['MACD'] > c['MACD_SIGNAL'] and c['MACD_HIST'] > 0:
        filters_met += 1
    
    # Filter 4: RSI in Optimal Zone (25-70, not overbought)
    if 25 < c['RSI_7'] < 70:
        filters_met += 1
    
    # Filter 5: Stochastic K > D (Momentum Up)
    if c['STOCH_K'] > c['STOCH_D']:
        filters_met += 1
    
    # Filter 6: Price bouncing from BB Lower (mean reversion)
    if c['close'] > c['BB_LOWER'] and p['close'] <= c['BB_LOWER']:
        filters_met += 1
    
    # Filter 7: ML Prediction Bullish
    if ml_pred == 1 and ml_proba > ML_CONFIDENCE_THRESHOLD:
        filters_met += 1
    
    # Signal jika >=5 filters met
    confidence = filters_met / 7.0
    return filters_met >= 5, confidence

# =============================================================================
# FUNGSI 5: DETECT SHORT SIGNAL (7 Filters)
# =============================================================================
def detect_short_signal(df, ml_pred, ml_proba):
    """Detect SHORT dengan 7 filter untuk 70%+ winrate"""
    if len(df) < 2:
        return False, 0
    
    c = df.iloc[-1]  # Current
    p = df.iloc[-2]  # Previous
    
    filters_met = 0
    
    # Filter 1: EMA Alignment (5 < 12 < 26)
    if c['EMA_5'] < c['EMA_12'] < c['EMA_26']:
        filters_met += 1
    
    # Filter 2: EMA Crossover (5 crossing below 12)
    if p['EMA_5'] >= p['EMA_12'] and c['EMA_5'] < c['EMA_12']:
        filters_met += 1
    
    # Filter 3: MACD Bearish (MACD < Signal)
    if c['MACD'] < c['MACD_SIGNAL'] and c['MACD_HIST'] < 0:
        filters_met += 1
    
    # Filter 4: RSI in Optimal Zone (30-75, not oversold)
    if 30 < c['RSI_7'] < 75:
        filters_met += 1
    
    # Filter 5: Stochastic K < D (Momentum Down)
    if c['STOCH_K'] < c['STOCH_D']:
        filters_met += 1
    
    # Filter 6: Price touching BB Upper (mean reversion)
    if c['close'] < c['BB_UPPER'] and p['close'] >= c['BB_UPPER']:
        filters_met += 1
    
    # Filter 7: ML Prediction Bearish
    if ml_pred == -1 and ml_proba > ML_CONFIDENCE_THRESHOLD:
        filters_met += 1
    
    # Signal jika >=5 filters met
    confidence = filters_met / 7.0
    return filters_met >= 5, confidence

# =============================================================================
# FUNGSI 6: FETCH DATA MULTI-TIMEFRAME
# =============================================================================
def fetch_multi_timeframe_data(symbol):
    """Fetch data dari 3 timeframe"""
    try:
        data_1m = pd.DataFrame(
            exchange.fetch_ohlcv(symbol, TIMEFRAME_PRIMARY, limit=300),
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        data_5m = pd.DataFrame(
            exchange.fetch_ohlcv(symbol, TIMEFRAME_CONFIRM, limit=100),
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        data_15m = pd.DataFrame(
            exchange.fetch_ohlcv(symbol, TIMEFRAME_TREND, limit=50),
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        return data_1m, data_5m, data_15m
    
    except Exception as e:
        Logger.error(f"Error fetching data {symbol}: {e}")
        return None, None, None

# =============================================================================
# FUNGSI 7: GET TREND (dari 5m & 15m)
# =============================================================================
def get_multi_timeframe_trend(df_5m, df_15m):
    """Determine trend dari multi-timeframe"""
    if len(df_5m) < 2 or len(df_15m) < 2:
        return "NEUTRAL", "NEUTRAL"
    
    # 5m Trend
    c_5m = df_5m.iloc[-1]
    trend_5m = "UP" if c_5m['EMA_5'] > c_5m['EMA_12'] > c_5m['EMA_26'] else \
               "DOWN" if c_5m['EMA_5'] < c_5m['EMA_12'] < c_5m['EMA_26'] else "NEUTRAL"
    
    # 15m Trend
    c_15m = df_15m.iloc[-1]
    trend_15m = "UP" if c_15m['EMA_5'] > c_15m['EMA_12'] > c_15m['EMA_26'] else \
                "DOWN" if c_15m['EMA_5'] < c_15m['EMA_12'] < c_15m['EMA_26'] else "NEUTRAL"
    
    return trend_5m, trend_15m

# =============================================================================
# FUNGSI 8: ANALYZE SYMBOL (MAIN)
# =============================================================================
def analyze_symbol(symbol):
    """Analisis lengkap symbol"""
    try:
        # Fetch data
        data_1m, data_5m, data_15m = fetch_multi_timeframe_data(symbol)
        if data_1m is None:
            return None
        
        # Calculate indicators
        df_1m = calculate_indicators_1m(data_1m)
        df_5m = calculate_indicators_1m(data_5m)
        df_15m = calculate_indicators_1m(data_15m)
        
        # Get trends
        trend_5m, trend_15m = get_multi_timeframe_trend(df_5m, df_15m)
        
        # Train ML
        ml_model = train_ml_model(df_1m)
        ml_pred, ml_proba = predict_direction(df_1m, ml_model)
        
        # Check LONG
        long_signal, long_confidence = detect_long_signal(df_1m, ml_pred, ml_proba)
        
        # Check SHORT
        short_signal, short_confidence = detect_short_signal(df_1m, ml_pred, ml_proba)
        
        current_price = df_1m['close'].iloc[-1]
        
        return {
            'symbol': symbol,
            'price': current_price,
            'long_signal': long_signal,
            'long_confidence': long_confidence,
            'short_signal': short_signal,
            'short_confidence': short_confidence,
            'trend_5m': trend_5m,
            'trend_15m': trend_15m,
            'ml_pred': ml_pred,
            'ml_proba': ml_proba,
            'df_1m': df_1m,
            'rsi': df_1m['RSI_7'].iloc[-1],
            'atr': df_1m['ATR'].iloc[-1]
        }
    
    except Exception as e:
        Logger.error(f"Error analyzing {symbol}: {e}")
        return None

# =============================================================================
# FUNGSI 9: CALCULATE DYNAMIC SL/TP
# =============================================================================
def calculate_sl_tp(symbol, direction, entry_price, atr_value):
    """Calculate SL & TP with ATR"""
    try:
        if direction == "LONG":
            sl = entry_price - (atr_value * ATR_SL_MULTIPLIER)
            tp = entry_price + (atr_value * ATR_TP_MULTIPLIER)
        else:  # SHORT
            sl = entry_price + (atr_value * ATR_SL_MULTIPLIER)
            tp = entry_price - (atr_value * ATR_TP_MULTIPLIER)
        
        # Precision
        sl = float(exchange.price_to_precision(symbol, sl))
        tp = float(exchange.price_to_precision(symbol, tp))
        
        return sl, tp
    
    except Exception as e:
        Logger.error(f"Error calculating SL/TP: {e}")
        return None, None

# =============================================================================
# FUNGSI 10: EXECUTE TRADE
# =============================================================================
def execute_trade(symbol, direction, entry_price, atr_value):
    """Execute trade dengan SL/TP otomatis"""
    try:
        Logger.signal(direction, symbol, 1.0, entry_price)
        
        # Set leverage
        exchange.set_leverage(LEVERAGE, symbol)
        time.sleep(0.5)
        
        # Calculate position
        amount = POSITION_SIZE_USDT / entry_price
        amount = float(exchange.amount_to_precision(symbol, amount))
        
        # Market order
        side = 'buy' if direction == "LONG" else 'sell'
        tp_side = 'sell' if direction == "LONG" else 'buy'
        
        Logger.info(f"Executing market order: {side} {amount} {symbol} @ ${entry_price:.8f}")
        order = exchange.create_order(symbol, 'market', side, amount)
        Logger.success(f"Market order executed: {order['id']}")
        
        time.sleep(1.5)
        
        # Calculate SL/TP
        sl_price, tp_price = calculate_sl_tp(symbol, direction, entry_price, atr_value)
        if sl_price is None:
            return False
        
        Logger.info(f"Setting SL: ${sl_price:.8f} | TP: ${tp_price:.8f}")
        
        # SL order
        sl_params = {'stopPrice': sl_price, 'reduceOnly': True}
        sl_order = exchange.create_order(
            symbol, 'STOP_MARKET', tp_side, amount, params=sl_params
        )
        Logger.success(f"SL order set: {sl_order['id']}")
        
        time.sleep(0.5)
        
        # TP order
        tp_params = {'stopPrice': tp_price, 'reduceOnly': True}
        tp_order = exchange.create_order(
            symbol, 'TAKE_PROFIT_MARKET', tp_side, amount, params=tp_params
        )
        Logger.success(f"TP order set: {tp_order['id']}")
        
        Logger.success(f"✅ Position {direction} for {symbol} fully configured!")
        return True
    
    except Exception as e:
        Logger.error(f"Error executing trade: {e}")
        return False

# =============================================================================
# FUNGSI 11: SCAN MARKETS
# =============================================================================
def scan_high_volume_symbols():
    """Scan symbols dengan volume tinggi"""
    try:
        Logger.info("Scanning high-volume symbols...")
        
        markets = exchange.load_markets()
        symbols = [
            m['symbol'] for m in markets.values()
            if m.get('swap') and m.get('quote') == 'USDT' and m.get('active')
        ]
        
        tradeable = []
        for symbol in symbols[:100]:  # Top 100
            try:
                ticker = exchange.fetch_ticker(symbol)
                volume_24h = ticker.get('quoteVolume', 0)
                
                if volume_24h > MIN_VOLUME_USDT:
                    tradeable.append(symbol)
                
                time.sleep(exchange.rateLimit / 1000)
            except:
                continue
        
        Logger.info(f"Found {len(tradeable)} tradeable symbols")
        return tradeable
    
    except Exception as e:
        Logger.error(f"Error scanning markets: {e}")
        return []

# =============================================================================
# MAIN BOT LOOP
# =============================================================================
if __name__ == '__main__':
    print("\n" + "="*80)
    print("🤖 ADVANCED 1-MINUTE SCALPING BOT v3.0".center(80))
    print("Target: 70%+ Winrate | Timeframe: 1-Minute".center(80))
    print("="*80 + "\n")
    
    Logger.success("Bot initialized!")
    Logger.info(f"Config: Leverage {LEVERAGE}x | Position Size: ${POSITION_SIZE_USDT} | Max Positions: {MAX_POSITIONS}")
    
    while True:
        try:
            Logger.info("\n" + "="*80)
            Logger.info(f"Scan Cycle: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Get current positions
            open_positions = exchange.fetch_positions()
            active_symbols = {
                p['info']['symbol'] for p in open_positions
                if float(p['info']['positionAmt']) != 0
            }
            
            current_pos_count = len(active_symbols)
            Logger.info(f"Active Positions: {current_pos_count}/{MAX_POSITIONS}")
            
            # Can open new positions?
            if current_pos_count < MAX_POSITIONS:
                # Scan markets
                symbols = scan_high_volume_symbols()
                
                for symbol in symbols:
                    if symbol in active_symbols:
                        continue  # Skip if already open
                    
                    if len(active_symbols) >= MAX_POSITIONS:
                        break  # Max reached
                    
                    # Analyze
                    analysis = analyze_symbol(symbol)
                    if analysis is None:
                        continue
                    
                    # Check LONG
                    if analysis['long_signal'] and analysis['long_confidence'] > 0.65:
                        if analysis['trend_5m'] in ["UP", "NEUTRAL"]:  # Trend confirmation
                            Logger.signal("LONG", symbol, analysis['long_confidence'], analysis['price'])
                            if execute_trade(symbol, "LONG", analysis['price'], analysis['atr']):
                                active_symbols.add(symbol)
                    
                    # Check SHORT
                    elif analysis['short_signal'] and analysis['short_confidence'] > 0.65:
                        if analysis['trend_5m'] in ["DOWN", "NEUTRAL"]:  # Trend confirmation
                            Logger.signal("SHORT", symbol, analysis['short_confidence'], analysis['price'])
                            if execute_trade(symbol, "SHORT", analysis['price'], analysis['atr']):
                                active_symbols.add(symbol)
                    
                    time.sleep(0.5)
            
            else:
                Logger.warning(f"All {MAX_POSITIONS} positions open. Waiting for exit...")
        
        except KeyboardInterrupt:
            Logger.warning("Bot stopped by user")
            break
        
        except Exception as e:
            Logger.error(f"Main loop error: {e}")
        
        finally:
            Logger.info(f"Next scan in {SCAN_INTERVAL}s...")
            time.sleep(SCAN_INTERVAL)

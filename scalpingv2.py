import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import warnings
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from collections import deque
import ta as technical_analysis

warnings.filterwarnings("ignore")

# --- PENGATURAN KONEKSI & API ---
BINANCE_API_KEY = 'GANTI_DENGAN_API_KEY_ANDA'
BINANCE_SECRET_KEY = 'GANTI_DENGAN_SECRET_KEY_ANDA'

exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET_KEY,
    'options': {
        'defaultType': 'future',
    },
})

# --- PENGATURAN STRATEGI & RISIKO YANG DIOPTIMALKAN ---
TIMEFRAME = '5m'
EMA_FAST_PERIOD = 5
EMA_SLOW_PERIOD = 20
EMA_TREND_PERIOD = 50
RSI_PERIOD = 7
RSI_OVERBOUGHT = 75
RSI_OVERSOLD = 25
POSITION_SIZE_USDT = 500
LEVERAGE = 10
ATR_SL_MULTIPLIER = 0.8
ATR_TP_MULTIPLIER = 1.5
SCAN_INTERVAL_SECONDS = 60
MAX_POSITIONS = 2

# --- PENGATURAN AI YANG DIOPTIMALKAN ---
ML_TRAINING_PERIOD = 200
ML_PREDICTION_THRESHOLD = 0.75
ENSEMBLE_MIN_VOTES = 2

# --- MODEL ENSEMBLE AI ---
class AdvancedTradingAI:
    def __init__(self):
        self.models = {
            'xgb': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
            'gbc': GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42),
            'rf': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.lstm_model = self._build_lstm_model()
        self.feature_importance = {}
        
    def _build_lstm_model(self):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(10, 25)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def create_advanced_features(self, df):
        """Membuat fitur teknikal yang lebih canggih untuk AI"""
        # Price-based features
        df['price_change_1'] = df['close'].pct_change(1)
        df['price_change_3'] = df['close'].pct_change(3)
        df['price_change_5'] = df['close'].pct_change(5)
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Volume features
        df['volume_sma_10'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_10']
        df['volume_price_trend'] = (df['volume'] * df['close']).pct_change(5)
        
        # Momentum indicators
        df['rsi_7'] = ta.rsi(df['close'], length=7)
        df['rsi_14'] = ta.rsi(df['close'], length=14)
        df['stoch_rsi'] = ta.stochrsi(df['close'], length=14)
        df['williams_r'] = technical_analysis.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
        df['cci'] = technical_analysis.trend.cci(df['high'], df['low'], df['close'], window=20)
        df['awesome_oscillator'] = technical_analysis.momentum.ao(df['high'], df['low'])
        
        # Trend indicators
        df['ema_5'] = ta.ema(df['close'], length=5)
        df['ema_20'] = ta.ema(df['close'], length=20)
        df['ema_50'] = ta.ema(df['close'], length=50)
        df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['macd'] = ta.macd(df['close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
        df['macd_signal'] = ta.macd(df['close'], fast=12, slow=26, signal=9)['MACDs_12_26_9']
        df['ichimoku_a'] = technical_analysis.trend.ichimoku_a(df['high'], df['low'], window1=9, window2=26)
        df['ichimoku_b'] = technical_analysis.trend.ichimoku_b(df['high'], df['low'], window2=26, window3=52)
        
        # Volatility indicators
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['bollinger_upper'] = ta.bbands(df['close'], length=20, std=2)['BBU_20_2.0']
        df['bollinger_lower'] = ta.bbands(df['close'], length=20, std=2)['BBL_20_2.0']
        df['bollinger_middle'] = ta.bbands(df['close'], length=20, std=2)['BBM_20_2.0']
        df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['bollinger_middle']
        df['keltner_upper'] = technical_analysis.volatility.keltner_channel_hband(df['high'], df['low'], df['close'], window=20)
        df['keltner_lower'] = technical_analysis.volatility.keltner_channel_lband(df['high'], df['low'], df['close'], window=20)
        
        # Support Resistance
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()
        df['distance_to_resistance'] = (df['resistance'] - df['close']) / df['close']
        df['distance_to_support'] = (df['close'] - df['support']) / df['close']
        
        # Pattern Recognition
        df['hammer'] = ((df['low'] - df['open']) > (df['high'] - df['low']) * 0.6) & (df['close'] > df['open'])
        df['shooting_star'] = ((df['high'] - df['close']) > (df['high'] - df['low']) * 0.6) & (df['close'] < df['open'])
        
        # Market Regime
        df['volatility_regime'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        df['trend_strength'] = abs(df['close'].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]))
        
        return df

    def prepare_features(self, df):
        """Mempersiapkan fitur untuk training AI"""
        feature_columns = [
            'price_change_1', 'price_change_3', 'price_change_5', 'high_low_ratio', 'close_open_ratio',
            'volume_ratio', 'volume_price_trend', 'rsi_7', 'rsi_14', 'stoch_rsi', 'williams_r', 'cci',
            'awesome_oscillator', 'adx', 'macd', 'macd_signal', 'atr', 'bollinger_width',
            'distance_to_resistance', 'distance_to_support', 'volatility_regime', 'trend_strength'
        ]
        
        # Hapus kolom yang tidak ada
        available_features = [col for col in feature_columns if col in df.columns]
        X = df[available_features].fillna(0).replace([np.inf, -np.inf], 0)
        
        return X, available_features

    def calculate_target(self, df, lookahead=3, threshold=0.008):
        """Menghitung target dengan threshold yang lebih baik"""
        future_returns = df['close'].shift(-lookahead) / df['close'] - 1
        
        # Multi-class target
        conditions = [
            future_returns > threshold,  # UP
            future_returns < -threshold, # DOWN
            True                         # NEUTRAL
        ]
        choices = [1, -1, 0]
        
        return np.select(conditions, choices)

    def train_ensemble(self, df):
        """Training ensemble model dengan data terbaru"""
        df = self.create_advanced_features(df)
        X, features = self.prepare_features(df)
        y = self.calculate_target(df)
        
        if len(X) < ML_TRAINING_PERIOD:
            return False
            
        # Gunakan data terbaru untuk training
        train_data = df.tail(ML_TRAINING_PERIOD)
        X_train = X.iloc[-ML_TRAINING_PERIOD:]
        y_train = y[-ML_TRAINING_PERIOD:]
        
        # Hapus samples dengan target NaN
        valid_idx = ~pd.isna(y_train)
        X_train = X_train[valid_idx]
        y_train = y_train[valid_idx]
        
        if len(X_train) < 100:
            return False
            
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train semua model
        for name, model in self.models.items():
            model.fit(X_scaled, y_train)
            
        # Hitung feature importance dari Random Forest
        self.feature_importance = dict(zip(features, self.models['rf'].feature_importances_))
        
        return True

    def ensemble_predict(self, df):
        """Prediksi menggunakan ensemble voting"""
        df = self.create_advanced_features(df)
        X, features = self.prepare_features(df)
        
        if len(X) == 0:
            return 0, 0.0
            
        current_features = X.iloc[-1:].values
        current_scaled = self.scaler.transform(current_features)
        
        predictions = []
        probabilities = []
        
        for name, model in self.models.items():
            pred = model.predict(current_scaled)[0]
            proba = model.predict_proba(current_scaled)[0].max()
            predictions.append(pred)
            probabilities.append(proba)
        
        # Ensemble voting dengan confidence
        final_pred = max(set(predictions), key=predictions.count)
        avg_confidence = np.mean(probabilities)
        
        return final_pred, avg_confidence

# Inisialisasi AI
trading_ai = AdvancedTradingAI()

# --- FUNGSI UTAMA YANG DIOPTIMALKAN ---
def scan_markets_improved():
    """Market scanner yang lebih selektif"""
    print(f"[{time.ctime()}] Memindai pasar dengan kriteria ketat...")
    potential_symbols = []
    
    try:
        all_markets = exchange.load_markets()
        symbols = [market['symbol'] for market in all_markets.values() 
                   if market.get('swap') and market.get('quote') == 'USDT' and market.get('active')]
        
        for symbol in symbols[:50]:  # Batasi untuk efisiensi
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, '15m', limit=100)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Filter volume lebih ketat
                volume_24h = df['volume'].tail(24).sum() * df['close'].mean()
                if volume_24h < 5_000_000:
                    continue
                
                # Filter trending dan momentum
                df['ema_20'] = ta.ema(df['close'], length=20)
                df['ema_50'] = ta.ema(df['close'], length=50)
                df['rsi'] = ta.rsi(df['close'], length=14)
                df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)
                
                current = df.iloc[-1]
                
                # Hanya simbol dengan trend kuat
                if current['adx'] < 25:
                    continue
                    
                # Harga di atas EMA 20 dan 50 (uptrend) atau di bawah (downtrend)
                price_above_ema20 = current['close'] > current['ema_20']
                price_above_ema50 = current['close'] > current['ema_50']
                
                if (price_above_ema20 and price_above_ema50) or (not price_above_ema20 and not price_above_ema50):
                    potential_symbols.append(symbol)
                    print(f"✓ {symbol} - ADX: {current['adx']:.1f}, Volume: {volume_24h:,.0f}")
                
                time.sleep(0.1)
                
            except Exception as e:
                continue
                
    except Exception as e:
        print(f"Error saat memindai pasar: {e}")
        
    return potential_symbols

def get_enhanced_signal(symbol, timeframe):
    """Sinyal trading yang ditingkatkan dengan multiple confirmation"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=300)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Training AI dengan data terbaru
        ai_trained = trading_ai.train_ensemble(df)
        
        if not ai_trained:
            return "NEUTRAL"
        
        # Dapatkan prediksi AI
        ai_prediction, ai_confidence = trading_ai.ensemble_predict(df)
        
        if ai_confidence < ML_PREDICTION_THRESHOLD:
            return "NEUTRAL"
        
        # Analisis teknikal konvensional sebagai konfirmasi
        df = trading_ai.create_advanced_features(df)
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Konfirmasi 1: Trend (EMA Alignment)
        ema_bullish = current['ema_5'] > current['ema_20'] > current['ema_50']
        ema_bearish = current['ema_5'] < current['ema_20'] < current['ema_50']
        
        # Konfirmasi 2: Momentum (RSI + Stochastic)
        rsi_ok = 30 < current['rsi_7'] < 70
        stoch_ok = 20 < current['stoch_rsi'] < 80
        
        # Konfirmasi 3: Volatility (Bollinger Band Position)
        bb_position = (current['close'] - current['bollinger_lower']) / (current['bollinger_upper'] - current['bollinger_lower'])
        bb_bullish = bb_position < 0.3  # Dekat lower band
        bb_bearish = bb_position > 0.7  # Dekat upper band
        
        # Konfirmasi 4: Volume confirmation
        volume_confirm = current['volume_ratio'] > 1.2
        
        # Konfirmasi 5: MACD Signal
        macd_bullish = current['macd'] > current['macd_signal'] and current['macd'] > 0
        macd_bearish = current['macd'] < current['macd_signal'] and current['macd'] < 0
        
        # Decision Making dengan multiple confirmation
        bull_confirmations = sum([ema_bullish, rsi_ok, stoch_ok, bb_bullish, volume_confirm, macd_bullish])
        bear_confirmations = sum([ema_bearish, rsi_ok, stoch_ok, bb_bearish, volume_confirm, macd_bearish])
        
        MIN_CONFIRMATIONS = 4
        
        if ai_prediction == 1 and bull_confirmations >= MIN_CONFIRMATIONS:
            return "LONG"
        elif ai_prediction == -1 and bear_confirmations >= MIN_CONFIRMATIONS:
            return "SHORT"
        else:
            return "NEUTRAL"
            
    except Exception as e:
        print(f"Error dalam analisis sinyal {symbol}: {e}")
        return "NEUTRAL"

def calculate_adaptive_sl_tp(symbol, timeframe, signal, df):
    """SL/TP adaptif berdasarkan volatilitas dan support/resistance"""
    try:
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        if signal == "LONG":
            # SL: dibawah support atau ATR-based, mana yang lebih baik
            support_level = df['support'].iloc[-1]
            sl_price = min(support_level, current_price - (atr * ATR_SL_MULTIPLIER))
            
            # TP: resistance atau ATR-based
            resistance_level = df['resistance'].iloc[-1]
            tp_price = min(resistance_level, current_price * 1.015)  # 1.5% target
            
        else:  # SHORT
            # SL: diatas resistance atau ATR-based
            resistance_level = df['resistance'].iloc[-1]
            sl_price = max(resistance_level, current_price + (atr * ATR_SL_MULTIPLIER))
            
            # TP: support atau ATR-based
            support_level = df['support'].iloc[-1]
            tp_price = max(support_level, current_price * 0.985)  # 1.5% target
            
        return sl_price, tp_price
        
    except Exception as e:
        print(f"Error menghitung SL/TP: {e}")
        # Fallback ke ATR sederhana
        atr = df['atr'].iloc[-1]
        if signal == "LONG":
            return current_price - (atr * ATR_SL_MULTIPLIER), current_price * 1.015
        else:
            return current_price + (atr * ATR_SL_MULTIPLIER), current_price * 0.985

def execute_trade_improved(symbol, signal, sl_price, tp_price):
    """Eksekusi trade dengan risk management yang lebih baik"""
    print(f"🚀 EKSEKUSI TRADE: {signal} {symbol}")
    
    try:
        # Set leverage
        exchange.set_leverage(LEVERAGE, symbol)
        
        # Get current price and calculate position size
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        amount = POSITION_SIZE_USDT / current_price
        
        # Precision
        market_info = exchange.market(symbol)
        amount = exchange.amount_to_precision(symbol, amount)
        
        # Entry order
        entry_side = 'buy' if signal == "LONG" else 'sell'
        print(f"1/3: Market Order {entry_side} {amount} {symbol}")
        exchange.create_order(symbol, 'market', entry_side, amount)
        time.sleep(1)
        
        # SL Order
        sl_side = 'sell' if signal == "LONG" else 'buy'
        sl_params = {'stopPrice': sl_price, 'reduceOnly': True}
        print(f"2/3: Stop Loss di {sl_price}")
        exchange.create_order(symbol, 'STOP_MARKET', sl_side, amount, None, sl_params)
        time.sleep(1)
        
        # TP Order  
        tp_params = {'stopPrice': tp_price, 'reduceOnly': True}
        print(f"3/3: Take Profit di {tp_price}")
        exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', sl_side, amount, None, tp_params)
        
        print(f"✅ TRADE BERHASIL: {signal} {symbol} | SL: {sl_price:.4f} | TP: {tp_price:.4f}")
        
    except Exception as e:
        print(f"❌ GAGAL EKSEKUSI TRADE: {e}")

# --- TRACKING PERFORMANCE ---
class PerformanceTracker:
    def __init__(self):
        self.trades = []
        self.win_count = 0
        self.loss_count = 0
        
    def add_trade(self, symbol, signal, entry_price, exit_price, pnl, win):
        trade = {
            'timestamp': time.time(),
            'symbol': symbol,
            'signal': signal,
            'entry': entry_price,
            'exit': exit_price,
            'pnl': pnl,
            'win': win
        }
        self.trades.append(trade)
        
        if win:
            self.win_count += 1
        else:
            self.loss_count += 1
            
    def get_winrate(self):
        total = self.win_count + self.loss_count
        return (self.win_count / total * 100) if total > 0 else 0
        
    def print_performance(self):
        winrate = self.get_winrate()
        print(f"\n📊 PERFORMANCE: {winrate:.1f}% Winrate ({self.win_count}/{self.win_count + self.loss_count})")
        
tracker = PerformanceTracker()

# --- MAIN LOOP YANG DIOPTIMALKAN ---
if __name__ == '__main__':
    print("""
    🤖 AI TRADING BOT - ENHANCED VERSION
    🔥 Target Winrate: 70-80%
    📈 Strategy: Multi-Timeframe + Ensemble AI + Advanced Technicals
    ⚠️  Risk Management: Adaptive SL/TP + Position Sizing
    """)
    
    last_scan_time = 0
    scan_interval = 120  # 2 menit untuk scan pasar
    
    while True:
        try:
            current_time = time.time()
            
            # Scan pasar setiap 2 menit
            if current_time - last_scan_time > scan_interval:
                print(f"\n[{time.ctime()}] Memulai scan pasar...")
                potential_symbols = scan_markets_improved()
                print(f"Ditemukan {len(potential_symbols)} simbol potensial")
                last_scan_time = current_time
            
            # Cek posisi aktif
            open_positions = exchange.fetch_positions()
            active_positions = [p for p in open_positions if float(p['info']['positionAmt']) != 0]
            
            print(f"Posisi aktif: {len(active_positions)}/{MAX_POSITIONS}")
            
            # Jika ada slot kosong, cari trading opportunity
            if len(active_positions) < MAX_POSITIONS and potential_symbols:
                for symbol in potential_symbols:
                    # Skip jika sudah ada posisi di symbol ini
                    if any(pos['symbol'] == symbol for pos in active_positions):
                        continue
                    
                    print(f"Analisis {symbol}...")
                    signal = get_enhanced_signal(symbol, TIMEFRAME)
                    
                    if signal in ["LONG", "SHORT"]:
                        # Dapatkan data terbaru untuk SL/TP calculation
                        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=100)
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df = trading_ai.create_advanced_features(df)
                        
                        sl_price, tp_price = calculate_adaptive_sl_tp(symbol, TIMEFRAME, signal, df)
                        
                        # Risk-Reward ratio check
                        current_price = df['close'].iloc[-1]
                        if signal == "LONG":
                            risk_reward = (tp_price - current_price) / (current_price - sl_price)
                        else:
                            risk_reward = (current_price - tp_price) / (sl_price - current_price)
                            
                        if risk_reward >= 1.5:  # Minimal 1:1.5 risk-reward
                            execute_trade_improved(symbol, signal, sl_price, tp_price)
                            
                            # Update active positions
                            active_positions = exchange.fetch_positions()
                            if len(active_positions) >= MAX_POSITIONS:
                                break
                        else:
                            print(f"Risk-Reward tidak memadai: {risk_reward:.2f}")
                    else:
                        print(f"Sinyal {symbol}: {signal}")
            
            # Print performance setiap 10 menit
            if int(time.time()) % 600 == 0:
                tracker.print_performance()
                
            time.sleep(SCAN_INTERVAL_SECONDS)
            
        except Exception as e:
            print(f"Error di main loop: {e}")
            time.sleep(30)

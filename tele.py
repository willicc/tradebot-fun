import ccxt
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import warnings
import requests
warnings.filterwarnings("ignore")

# --- PENGATURAN KONEKSI & API ---
# Ganti dengan API Key dan Secret Key Binance Anda
BINANCE_API_KEY = 'GANTI_DENGAN_API_KEY_ANDA'
BINANCE_SECRET_KEY = 'GANTI_DENGAN_SECRET_KEY_ANDA'

# --- PENGATURAN TELEGRAM ---
TELEGRAM_BOT_TOKEN = 'GANTI_DENGAN_BOT_TOKEN_TELEGRAM_ANDA'
TELEGRAM_CHAT_ID = 'GANTI_DENGAN_CHAT_ID_TELEGRAM_ANDA'

# Inisialisasi koneksi ke Binance Futures
exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET_KEY,
    'options': {
        'defaultType': 'future',
    },
})

# --- PENGATURAN STRATEGI & RISIKO ---
TIMEFRAME = '15m'
# Pengaturan EMA
EMA_FAST_PERIOD = 12
EMA_SLOW_PERIOD = 26
# Pengaturan RSI
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_MIDLINE = 50
# Pengaturan Posisi & Risiko
POSITION_SIZE_USDT = 150
LEVERAGE = 10
ATR_SL_MULTIPLIER = 1.0  # Lebih ketat
ATR_TP_MULTIPLIER = 1.5   # Risk-reward rendah karena winrate tinggi
# Pengaturan Bot
SCAN_INTERVAL_SECONDS = 300  # 5 menit
# Pengaturan AI
ML_TRAINING_PERIOD = 200  # Jumlah candle untuk training
ML_PREDICTION_THRESHOLD = 0.65  # Minimal probabilitas untuk sinyal

# ==============================================================================
# FUNGSI 1: KIRIM NOTIFIKASI KE TELEGRAM
# ==============================================================================
def send_telegram_message(message):
    """Mengirim pesan ke Telegram."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print(f"❌ Gagal mengirim pesan ke Telegram: {response.text}")
        else:
            print("✅ Pesan Telegram terkirim!")
    except Exception as e:
        print(f"❌ Error saat mengirim pesan Telegram: {e}")

# ==============================================================================
# FUNGSI 2: MARKET SCANNER (Lebih ketat)
# ==============================================================================
def scan_markets():
    """Memindai pasar untuk menemukan token dengan volatilitas tinggi dan trending."""
    print(f"[{time.ctime()}] Memindai pasar Binance Futures...")
    potential_symbols = []
    try:
        all_markets = exchange.load_markets()
        symbols = [market['symbol'] for market in all_markets.values() 
                   if market.get('swap') and market.get('quote') == 'USDT' and market.get('active')]
        
        for symbol in symbols:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=25)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                volume_24h_usdt = (df['close'] * df['volume']).sum()
                if volume_24h_usdt < 5_000_000:
                    continue

                # Tambahkan filter: trending dan tidak sideways
                df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
                df['ATR_PCT'] = (df['ATR'] / df['close']) * 100
                avg_atr_pct = df['ATR_PCT'].tail(5).mean()
                
                # Filter: tidak sideways (ATR > 1.5%)
                if avg_atr_pct < 1.5:
                    continue

                price_change_1h = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
                
                if abs(price_change_1h) > 2.5:
                    print(f"-> Token potensial ditemukan: {symbol} | Perubahan 1 Jam: {price_change_1h:.2f}% | ATR%: {avg_atr_pct:.2f}%")
                    potential_symbols.append(symbol)
                
                time.sleep(exchange.rateLimit / 1000)
            except Exception:
                continue 

    except Exception as e:
        print(f"Error saat memindai pasar: {e}")
        
    return list(set(potential_symbols))

# ==============================================================================
# FUNGSI 3: ANALISIS MULTI-INDIKATOR (Revisi Besar)
# ==============================================================================
def calculate_features(df):
    """Menghitung fitur-fitur teknikal untuk ML dan analisis lanjutan."""
    # EMA
    df.ta.ema(length=EMA_FAST_PERIOD, append=True)
    df.ta.ema(length=EMA_SLOW_PERIOD, append=True)
    
    # RSI
    df.ta.rsi(length=RSI_PERIOD, append=True)
    
    # MACD
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    
    # Bollinger Bands
    df.ta.bbands(length=20, std=2, append=True)
    
    # ATR
    df.ta.atr(length=14, append=True)
    
    # Stochastic RSI
    df.ta.stochrsi(length=14, append=True)
    
    # Volume Indicators
    df.ta.vwma(length=20, append=True)
    df.ta.obv(append=True)
    
    # ADX
    df.ta.adx(length=14, append=True)
    
    # Fitur tambahan
    df['price_change_pct'] = df['close'].pct_change()
    df['volatility'] = df['close'].rolling(10).std()
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Gap antara harga dan EMA
    df['ema_gap'] = (df[f'EMA_{EMA_FAST_PERIOD}'] - df[f'EMA_{EMA_SLOW_PERIOD}']) / df['close']
    
    return df

def get_ml_prediction(df):
    """Prediksi arah harga menggunakan model ML."""
    df = df.copy()
    df = calculate_features(df)
    
    # Ambil data terbaru
    df = df.dropna()
    if len(df) < ML_TRAINING_PERIOD + 20:
        return None, 0

    # Fitur untuk training
    features = [
        f'EMA_{EMA_FAST_PERIOD}', f'EMA_{EMA_SLOW_PERIOD}',
        f'RSI_{RSI_PERIOD}',
        f'MACD_{12}_{26}_{9}', f'MACD_{12}_{26}_{9}_S_9', f'MACD_{12}_{26}_{9}_H_9',
        f'BBL_{20}_2.0', f'BBM_{20}_2.0', f'BBU_{20}_2.0',
        f'STOCHRSI_{14}', f'STOCHRSI_{14}_K',
        'volatility', 'volume_ratio', 'ema_gap', 'ADX_14'
    ]
    
    df_ml = df[features].copy()
    df_ml['target'] = df['close'].shift(-1).pct_change().apply(lambda x: 1 if x > 0.001 else (-1 if x < -0.001 else 0))
    
    df_ml = df_ml.dropna()
    if len(df_ml) < ML_TRAINING_PERIOD:
        return None, 0

    # Ambil data training dan testing
    train_data = df_ml.tail(ML_TRAINING_PERIOD)
    X_train = train_data[features].values
    y_train = train_data['target'].values

    # Buat model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model.fit(X_train_scaled, y_train)

    # Prediksi untuk candle saat ini
    current_features = df[features].iloc[-1:].values
    current_features_scaled = scaler.transform(current_features)
    prediction = model.predict(current_features_scaled)[0]
    proba = model.predict_proba(current_features_scaled).max()

    return prediction, proba

def get_signal(symbol, timeframe):
    """Menganalisis data untuk sinyal LONG, SHORT, atau NEUTRAL dengan multi-filter."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = calculate_features(df)
        df = df.dropna()

        # Ambil baris terakhir
        current = df.iloc[-1]
        prev = df.iloc[-2]

        # Ambil ML prediction
        ml_pred, ml_proba = get_ml_prediction(df)
        if ml_pred is None or ml_proba < ML_PREDICTION_THRESHOLD:
            return "NEUTRAL"

        # Ambil indikator
        ema_fast = current[f'EMA_{EMA_FAST_PERIOD}']
        ema_slow = current[f'EMA_{EMA_SLOW_PERIOD}']
        prev_ema_fast = prev[f'EMA_{EMA_FAST_PERIOD}']
        prev_ema_slow = prev[f'EMA_{EMA_SLOW_PERIOD}']
        rsi = current[f'RSI_{RSI_PERIOD}']
        stoch_rsi_k = current[f'STOCHRSI_{14}_K']
        adx = current['ADX_14']
        macd_line = current[f'MACD_{12}_{26}_{9}']
        macd_signal = current[f'MACD_{12}_{26}_{9}_S_9']
        bb_lower = current[f'BBL_{20}_2.0']
        bb_upper = current[f'BBU_{20}_2.0']
        close = current['close']

        # Filter 1: Tren kuat (ADX > 25)
        if adx < 25:
            return "NEUTRAL"

        # Filter 2: RSI dan StochRSI tidak ekstrem
        if rsi > RSI_OVERBOUGHT or rsi < RSI_OVERSOLD:
            return "NEUTRAL"

        # Filter 3: Konfirmasi EMA crossover
        golden_cross = (prev_ema_fast <= prev_ema_slow) and (ema_fast > ema_slow)
        death_cross = (prev_ema_fast >= prev_ema_slow) and (ema_fast < ema_slow)

        # Filter 4: MACD konfirmasi
        macd_bullish = macd_line > macd_signal
        macd_bearish = macd_line < macd_signal

        # Filter 5: Harga di atas/bawah BB (mean reversion)
        price_near_lower_bb = close < bb_lower * 1.005
        price_near_upper_bb = close > bb_upper * 0.995

        # Kombinasi sinyal
        if ml_pred == 1 and golden_cross and macd_bullish and rsi < 60 and rsi > 40:
            return "LONG"
        elif ml_pred == -1 and death_cross and macd_bearish and rsi > 40 and rsi < 60:
            return "SHORT"
        else:
            return "NEUTRAL"

    except Exception as e:
        print(f"Error saat mengambil sinyal untuk {symbol}: {e}")
        return "ERROR"

# ==============================================================================
# FUNGSI 4 & 5 (KALKULASI SL/TP & EKSEKUSI TRADE)
# ==============================================================================
def calculate_dynamic_sl_tp(symbol, timeframe, signal):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.ta.atr(length=14, append=True)
        current_price = df['close'].iloc[-1]
        atr_value = df['ATRr_14'].iloc[-1]
        
        if signal == "LONG":
            sl_price = current_price - (atr_value * ATR_SL_MULTIPLIER)
            tp_price = current_price + (atr_value * ATR_TP_MULTIPLIER)
        else:
            sl_price = current_price + (atr_value * ATR_SL_MULTIPLIER)
            tp_price = current_price - (atr_value * ATR_TP_MULTIPLIER)
        
        return sl_price, tp_price
    except Exception as e:
        print(f"Error saat menghitung SL/TP dinamis untuk {symbol}: {e}")
        return None, None

def execute_trade(symbol, signal, sl_price, tp_price):
    print(f"MENCOBA EKSEKUSI TRADE: {signal} untuk {symbol}")
    try:
        exchange.set_leverage(LEVERAGE, symbol)
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        amount = POSITION_SIZE_USDT / current_price
        entry_side = 'buy' if signal == "LONG" else 'sell'
        sl_tp_side = 'sell' if signal == "LONG" else 'buy'
        market_info = exchange.market(symbol)
        amount = exchange.amount_to_precision(symbol, amount)
        sl_price = exchange.price_to_precision(symbol, sl_price)
        tp_price = exchange.price_to_precision(symbol, tp_price)

        print(f"1/3: Mengirim Market Order {entry_side} sebesar {amount} {market_info['base']}")
        exchange.create_order(symbol, 'market', entry_side, amount)
        time.sleep(2)
        print(f"2/3: Mengirim Stop Loss Order di harga {sl_price}")
        sl_params = {'stopPrice': sl_price, 'reduceOnly': True}
        exchange.create_order(symbol, 'STOP_MARKET', sl_tp_side, amount, params=sl_params)
        print(f"3/3: Mengirim Take Profit Order di harga {tp_price}")
        tp_params = {'stopPrice': tp_price, 'reduceOnly': True}
        exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', sl_tp_side, amount, params=tp_params)
        print(f"✅ ✅ ✅ SEMUA ORDER BERHASIL DIBUAT UNTUK POSISI BARU: {symbol} ✅ ✅ ✅")
        
        # Kirim notifikasi ke Telegram
        telegram_message = f"""
🚨 SINYAL DITEMUKAN 🚨
Pair: {symbol}
Aksi: {signal}
Harga Masuk: ${current_price:.4f}
SL: ${sl_price:.4f}
TP: ${tp_price:.4f}
Leverage: {LEVERAGE}x
Modal: ${POSITION_SIZE_USDT} USDT
Waktu: {time.ctime()}
        """
        send_telegram_message(telegram_message)
        
    except Exception as e:
        print(f"❌ ❌ ❌ GAGAL MEMBUAT ORDER LENGKAP untuk {symbol}: {e} ❌ ❌ ❌")

# ==============================================================================
# FUNGSI 6: BACKTESTING (Opsional untuk menguji winrate)
# ==============================================================================
def run_backtest(symbol, timeframe, days=30):
    """Menjalankan backtest untuk menghitung winrate historis."""
    print(f"\n--- BACKTESTING untuk {symbol} ---")
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=days * 96)  # 96 candle per day di 15m
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = calculate_features(df)
    
    signals = []
    for i in range(len(df)):
        if i < 100:
            signals.append("NEUTRAL")
            continue
        
        temp_df = df.iloc[:i+1].copy()
        signal = get_signal_from_df(temp_df, timeframe)
        signals.append(signal)

    df['signal'] = signals
    df['entry_price'] = np.nan
    df.loc[df['signal'] == 'LONG', 'entry_price'] = df['close']
    df.loc[df['signal'] == 'SHORT', 'entry_price'] = df['close']
    
    df['exit_price'] = df['close'].shift(-1)
    df['pnl'] = np.where(
        df['signal'] == 'LONG',
        (df['exit_price'] - df['entry_price']) / df['entry_price'],
        np.where(
            df['signal'] == 'SHORT',
            (df['entry_price'] - df['exit_price']) / df['entry_price'],
            0
        )
    )
    
    df['win'] = df['pnl'] > 0
    wins = df['win'].sum()
    total_trades = df['signal'].apply(lambda x: x in ['LONG', 'SHORT']).sum()
    
    if total_trades == 0:
        winrate = 0
    else:
        winrate = (wins / total_trades) * 100
    
    print(f"Total trades: {total_trades}")
    print(f"Winrate: {winrate:.2f}%")
    return winrate

def get_signal_from_df(df, timeframe):
    """Fungsi helper untuk backtesting."""
    try:
        df = calculate_features(df)
        df = df.dropna()

        current = df.iloc[-1]
        prev = df.iloc[-2]

        ema_fast = current[f'EMA_{EMA_FAST_PERIOD}']
        ema_slow = current[f'EMA_{EMA_SLOW_PERIOD}']
        prev_ema_fast = prev[f'EMA_{EMA_FAST_PERIOD}']
        prev_ema_slow = prev[f'EMA_{EMA_SLOW_PERIOD}']
        rsi = current[f'RSI_{RSI_PERIOD}']
        adx = current['ADX_14']

        if adx < 25:
            return "NEUTRAL"

        golden_cross = (prev_ema_fast <= prev_ema_slow) and (ema_fast > ema_slow)
        death_cross = (prev_ema_fast >= prev_ema_slow) and (ema_fast < ema_slow)

        if golden_cross and 40 < rsi < 60:
            return "LONG"
        elif death_cross and 40 < rsi < 60:
            return "SHORT"
        else:
            return "NEUTRAL"
    except:
        return "NEUTRAL"

# ==============================================================================
# LOOP UTAMA BOT
# ==============================================================================
if __name__ == '__main__':
    print("Bot Trading Revisi (Multi-Filter + ML + Telegram Notification) Dimulai...")
    print("Mode: Memindai semua pasar, membuka maks 1 posisi.")
    print("---")
    
    while True:
        try:
            open_positions = exchange.fetch_positions()
            active_position_symbols = {p['info']['symbol'] for p in open_positions if float(p['info']['positionAmt']) != 0}
            if not active_position_symbols:
                print(f"[{time.ctime()}] Tidak ada posisi terbuka. Mencari peluang...")
                potential_symbols = scan_markets()
                if not potential_symbols:
                    print("Tidak ada token potensial yang ditemukan saat ini.")
                for symbol in potential_symbols:
                    print(f"Menganalisis sinyal untuk {symbol}...")
                    signal = get_signal(symbol, TIMEFRAME)
                    if signal in ["LONG", "SHORT"]:
                        sl_price, tp_price = calculate_dynamic_sl_tp(symbol, TIMEFRAME, signal)
                        if sl_price is not None and tp_price is not None:
                            execute_trade(symbol, signal, sl_price, tp_price)
                            print(f"Posisi untuk {symbol} telah dibuka. Menghentikan pemindaian siklus ini.")
                            break 
                    else:
                        print(f"Sinyal untuk {symbol}: {signal}. Tidak ada aksi.")
            else:
                print(f"[{time.ctime()}] Posisi sudah terbuka untuk {active_position_symbols}. Menunggu posisi ditutup.")
        except Exception as e:
            print(f"Terjadi error besar di loop utama: {e}")
        finally:
            print(f"--- Siklus selesai. Menunggu {SCAN_INTERVAL_SECONDS} detik. ---")
            time.sleep(SCAN_INTERVAL_SECONDS)

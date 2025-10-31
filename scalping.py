import ccxt
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

# --- PENGATURAN KONEKSI & API ---
# Ganti dengan API Key dan Secret Key Binance Anda
BINANCE_API_KEY = 'GANTI_DENGAN_API_KEY_ANDA'
BINANCE_SECRET_KEY = 'GANTI_DENGAN_SECRET_KEY_ANDA'

# Inisialisasi koneksi ke Binance Futures
exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET_KEY,
    'options': {
        'defaultType': 'future',
    },
})

# --- PENGATURAN STRATEGI & RISIKO ---
TIMEFRAME = '5m'  # Lebih pendek untuk scalping
# Pengaturan EMA
EMA_FAST_PERIOD = 8
EMA_SLOW_PERIOD = 21
# Pengaturan RSI
RSI_PERIOD = 9
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_MIDLINE = 50
# Pengaturan Posisi & Risiko
POSITION_SIZE_USDT = 1000  # Lebih kecil untuk scalping
LEVERAGE = 10            # Lebih tinggi untuk scalping
ATR_SL_MULTIPLIER = 1.0  # Lebih ketat
ATR_TP_MULTIPLIER = 2.0  # Target profit 1% (lebih rendah dari ATR)
# Pengaturan Bot
SCAN_INTERVAL_SECONDS = 120  # 2 menit untuk scalping
# Pengaturan AI
ML_TRAINING_PERIOD = 100  # Lebih pendek karena timeframe 5m
ML_PREDICTION_THRESHOLD = 0.70  # Lebih ketat untuk scalping
# Pengaturan Multi-Posisi
MAX_POSITIONS = 3  # Jumlah maksimal posisi yang ingin dijaga

# ==============================================================================
# FUNGSI 1: MARKET SCANNER (Lebih ketat untuk scalping)
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
                ohlcv = exchange.fetch_ohlcv(symbol, '5m', limit=50)  # 5m
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                volume_24h_usdt = (df['close'] * df['volume']).sum()
                if volume_24h_usdt < 1_000_000:  # Lebih ketat
                    continue

                # Tambahkan filter: trending dan tidak sideways
                df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
                df['ATR_PCT'] = (df['ATR'] / df['close']) * 100
                avg_atr_pct = df['ATR_PCT'].tail(10).mean()
                
                # Filter: tidak sideways (ATR > 1.5%)
                if avg_atr_pct < 1.0:  # Lebih ketat
                    continue

                price_change_5m = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
                
                if abs(price_change_5m) > 1.0:  # Lebih sensitif
                    print(f"-> Token potensial ditemukan: {symbol} | Perubahan 5m: {price_change_5m:.2f}% | ATR%: {avg_atr_pct:.2f}%")
                    potential_symbols.append(symbol)
                
                time.sleep(exchange.rateLimit / 1000)
            except Exception:
                continue 

    except Exception as e:
        print(f"Error saat memindai pasar: {e}")
        
    return list(set(potential_symbols))

# ==============================================================================
# FUNGSI 2: ANALISIS MULTI-INDIKATOR (Ditambahkan untuk scalping)
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
    
    # Heikin Ashi
    df['HA_open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
    df['HA_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['HA_high'] = df[['high', 'HA_open', 'HA_close']].max(axis=1)
    df['HA_low'] = df[['low', 'HA_open', 'HA_close']].min(axis=1)
    
    # Momentum
    df['momentum'] = df['close'].pct_change(5)  # 5 candle momentum
    
    # Price Action: Doji
    df['body_size'] = abs(df['close'] - df['open'])
    df['total_range'] = df['high'] - df['low']
    df['is_doji'] = df['body_size'] < (df['total_range'] * 0.1)  # Doji jika body < 10% range
    
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
        'volatility', 'volume_ratio', 'ema_gap', 'ADX_14',
        'HA_open', 'HA_close', 'HA_high', 'HA_low',
        'momentum'
    ]
    
    df_ml = df[features].copy()
    df_ml['target'] = df['close'].shift(-1).pct_change().apply(lambda x: 1 if x > 0.005 else (-1 if x < -0.005 else 0))  # 0.5% target
    
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
        ha_close = current['HA_close']
        ha_open = current['HA_open']
        momentum = current['momentum']
        is_doji = current['is_doji']

        # Filter 1: Tren kuat (ADX > 20 untuk scalping)
        if adx < 20:
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
        price_near_lower_bb = close < bb_lower * 1.002
        price_near_upper_bb = close > bb_upper * 0.998

        # Filter 6: Heikin Ashi konfirmasi
        ha_bullish = ha_close > ha_open
        ha_bearish = ha_close < ha_open

        # Filter 7: Momentum positif/negatif
        mom_positive = momentum > 0.001
        mom_negative = momentum < -0.001

        # Filter 8: Tidak masuk saat Doji (tidak pasti)
        if is_doji:
            return "NEUTRAL"

        # Kombinasi sinyal
        if ml_pred == 1 and golden_cross and macd_bullish and ha_bullish and mom_positive and rsi < 60 and rsi > 40:
            return "LONG"
        elif ml_pred == -1 and death_cross and macd_bearish and ha_bearish and mom_negative and rsi > 40 and rsi < 60:
            return "SHORT"
        else:
            return "NEUTRAL"

    except Exception as e:
        print(f"Error saat mengambil sinyal untuk {symbol}: {e}")
        return "ERROR"

# ==============================================================================
# FUNGSI 3 & 4 (KALKULASI SL/TP & EKSEKUSI TRADE)
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
            tp_price = current_price * 1.01  # Target profit 1%
        else:
            sl_price = current_price + (atr_value * ATR_SL_MULTIPLIER)
            tp_price = current_price * 0.99  # Target profit 1%
        
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
    except Exception as e:
        print(f"❌ ❌ ❌ GAGAL MEMBUAT ORDER LENGKAP untuk {symbol}: {e} ❌ ❌ ❌")

# ==============================================================================
# FUNGSI 5: BACKTESTING (Opsional untuk menguji winrate)
# ==============================================================================
def run_backtest(symbol, timeframe, days=7):  # Kurangi ke 7 hari untuk scalping
    """Menjalankan backtest untuk menghitung winrate historis."""
    print(f"\n--- BACKTESTING untuk {symbol} ---")
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=days * 288)  # 288 candle per day di 5m
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = calculate_features(df)
    
    signals = []
    for i in range(len(df)):
        if i < 50:
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
    
    df['win'] = df['pnl'] > 0.005  # Lebih dari 0.5% profit
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

        if adx < 20:
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
# LOOP UTAMA BOT (DIUBAH UNTUK MULTI-POSI & AUTO-REFILL)
# ==============================================================================
if __name__ == '__main__':
    print(f"Bot Scalping 1% Profit (Multi-Filter + ML + Scalping Indicators) Dimulai...")
    print(f"Mode: Memindai semua pasar, menjaga maks {MAX_POSITIONS} posisi aktif.")
    print("---")
    
    while True:
        try:
            open_positions = exchange.fetch_positions()
            active_position_symbols = {p['info']['symbol'] for p in open_positions if float(p['info']['positionAmt']) != 0}
            
            # Hitung jumlah posisi aktif
            current_positions = len(active_position_symbols)
            print(f"[{time.ctime()}] Jumlah posisi saat ini: {current_positions}/{MAX_POSITIONS}")
            
            # Jika posisi < MAX_POSITIONS, cari peluang
            if current_positions < MAX_POSITIONS:
                print(f"Posisi aktif kurang dari {MAX_POSITIONS}, mencari peluang...")
                potential_symbols = scan_markets()
                if not potential_symbols:
                    print("Tidak ada token potensial yang ditemukan saat ini.")
                
                # Loop semua potential_symbols dan buka posisi jika sinyal valid
                for symbol in potential_symbols:
                    # Cek apakah symbol sudah ada posisi aktif
                    if symbol in active_position_symbols:
                        continue  # Lewati jika sudah ada posisi
                    
                    # Cek apakah jumlah posisi sudah mencapai batas
                    if len(active_position_symbols) >= MAX_POSITIONS:
                        print(f"Jumlah posisi sudah mencapai batas maksimal ({MAX_POSITIONS}). Menghentikan pemindaian siklus ini.")
                        break

                    print(f"Menganalisis sinyal untuk {symbol}...")
                    signal = get_signal(symbol, TIMEFRAME)
                    if signal in ["LONG", "SHORT"]:
                        sl_price, tp_price = calculate_dynamic_sl_tp(symbol, TIMEFRAME, signal)
                        if sl_price is not None and tp_price is not None:
                            execute_trade(symbol, signal, sl_price, tp_price)
                            print(f"Posisi untuk {symbol} telah dibuka.")
                            active_position_symbols.add(symbol)  # Tambahkan ke set
                            
                            # Cek apakah sudah mencapai batas maksimal
                            if len(active_position_symbols) >= MAX_POSITIONS:
                                print(f"Jumlah posisi sudah mencapai batas maksimal ({MAX_POSITIONS}). Menghentikan pemindaian siklus ini.")
                                break
                    else:
                        print(f"Sinyal untuk {symbol}: {signal}. Tidak ada aksi.")
            else:
                print(f"[{time.ctime()}] Semua {MAX_POSITIONS} posisi sudah terbuka: {active_position_symbols}. Menunggu posisi ditutup.")
                
        except Exception as e:
            print(f"Terjadi error besar di loop utama: {e}")
        finally:
            print(f"--- Siklus selesai. Menunggu {SCAN_INTERVAL_SECONDS} detik. ---")
            time.sleep(SCAN_INTERVAL_SECONDS)

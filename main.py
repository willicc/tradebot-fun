import ccxt
import pandas as pd
import pandas_ta as ta
import time

# --- PENGATURAN AWAL ---
# Ganti dengan API Key dan Secret Key Binance Anda yang sudah diaktifkan untuk Futures
BINANCE_API_KEY = 'GANTI_DENGAN_API_KEY_ANDA'
BINANCE_SECRET_KEY = 'GANTI_DENGAN_SECRET_KEY_ANDA'

# Inisialisasi koneksi ke Binance Futures
exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET_KEY,
    'options': {
        'defaultType': 'future',  # Wajib untuk trading futures/swap USDⓈ-M
    },
})

# --- PENGATURAN STRATEGI & RISIKO ---
TIMEFRAME = '15m'         # Timeframe candle (e.g., 5m, 15m, 1h)
EMA_FAST_PERIOD = 10      # Periode EMA cepat
EMA_SLOW_PERIOD = 30      # Periode EMA lambat
POSITION_SIZE_USDT = 15   # Ukuran posisi dalam USDT (modal x leverage)
LEVERAGE = 10             # Leverage yang akan digunakan
ATR_SL_MULTIPLIER = 1.5   # Multiplier ATR untuk Stop Loss
ATR_TP_MULTIPLIER = 3.0   # Multiplier ATR untuk Take Profit (Risk/Reward 1:2)
SCAN_INTERVAL_SECONDS = 300 # Seberapa sering bot memindai pasar (detik), misal 300 = 5 menit

# ==============================================================================
# FUNGSI 1: MARKET SCANNER
# ==============================================================================
def scan_markets():
    """Memindai pasar untuk menemukan token yang paling volatil."""
    print(f"[{time.ctime()}] Memindai pasar Binance Futures...")
    potential_symbols = []
    try:
        all_markets = exchange.load_markets()
        symbols = [market['symbol'] for market in all_markets.values() 
                   if market.get('swap') and market.get('quote') == 'USDT' and market.get('active')]
        
        for symbol in symbols:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=25) # Ambil data 24 jam terakhir
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Filter berdasarkan volume (misal > $5 juta dalam 24 jam)
                volume_24h_usdt = (df['close'] * df['volume']).sum()
                if volume_24h_usdt < 5_000_000:
                    continue

                # Hitung perubahan harga dalam 1 jam terakhir
                price_change_1h = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
                
                if abs(price_change_1h) > 2.5: # Cari token dengan perubahan > 2.5% dalam 1 jam
                    print(f"-> Token potensial ditemukan: {symbol} | Perubahan 1 Jam: {price_change_1h:.2f}% | Volume 24 Jam: ${volume_24h_usdt:,.0f}")
                    potential_symbols.append(symbol)
                
                time.sleep(exchange.rateLimit / 1000) # Hormati rate limit API
            except Exception:
                continue 

    except Exception as e:
        print(f"Error saat memindai pasar: {e}")
        
    return list(set(potential_symbols)) 

# ==============================================================================
# FUNGSI 2: GENERATOR SINYAL (EMA CROSSOVER)
# ==============================================================================
def get_signal(symbol, timeframe):
    """Menganalisis data untuk sinyal LONG, SHORT, atau NEUTRAL."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.ta.ema(length=EMA_FAST_PERIOD, append=True)
        df.ta.ema(length=EMA_SLOW_PERIOD, append=True)
        
        last_row = df.iloc[-2]
        current_row = df.iloc[-1]

        if last_row[f'EMA_{EMA_FAST_PERIOD}'] < last_row[f'EMA_{EMA_SLOW_PERIOD}'] and \
           current_row[f'EMA_{EMA_FAST_PERIOD}'] > current_row[f'EMA_{EMA_SLOW_PERIOD}']:
            return "LONG"
        
        elif last_row[f'EMA_{EMA_FAST_PERIOD}'] > last_row[f'EMA_{EMA_SLOW_PERIOD}'] and \
             current_row[f'EMA_{EMA_FAST_PERIOD}'] < current_row[f'EMA_{EMA_SLOW_PERIOD}']:
            return "SHORT"
            
        else:
            return "NEUTRAL"

    except Exception as e:
        print(f"Error saat mengambil sinyal untuk {symbol}: {e}")
        return "ERROR"

# ==============================================================================
# FUNGSI 3: KALKULASI SL/TP DINAMIS (ATR)
# ==============================================================================
def calculate_dynamic_sl_tp(symbol, timeframe, signal):
    """Menghitung harga SL dan TP berdasarkan ATR."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.ta.atr(length=14, append=True)
        
        current_price = df['close'].iloc[-1]
        atr_value = df['ATRr_14'].iloc[-1]
        
        if signal == "LONG":
            sl_price = current_price - (atr_value * ATR_SL_MULTIPLIER)
            tp_price = current_price + (atr_value * ATR_TP_MULTIPLIER)
        else: # SHORT
            sl_price = current_price + (atr_value * ATR_SL_MULTIPLIER)
            tp_price = current_price - (atr_value * ATR_TP_MULTIPLIER)
            
        return sl_price, tp_price

    except Exception as e:
        print(f"Error saat menghitung SL/TP dinamis untuk {symbol}: {e}")
        return None, None

# ==============================================================================
# FUNGSI 4: EKSEKUSI TRADE
# ==============================================================================
def execute_trade(symbol, signal, sl_price, tp_price):
    """Mengeksekusi trade dengan mengirim order entry, lalu SL dan TP."""
    print(f"MENCOBA EKSEKUSI TRADE: {signal} untuk {symbol}")
    try:
        exchange.set_leverage(LEVERAGE, symbol)
        
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        amount = POSITION_SIZE_USDT / current_price
        
        entry_side = 'buy' if signal == "LONG" else 'sell'
        sl_tp_side = 'sell' if signal == "LONG" else 'buy'
        
        # Dapatkan info market untuk pembulatan presisi
        market_info = exchange.market(symbol)
        amount = exchange.amount_to_precision(symbol, amount)
        sl_price = exchange.price_to_precision(symbol, sl_price)
        tp_price = exchange.price_to_precision(symbol, tp_price)

        # 1. Kirim order masuk (MARKET order)
        print(f"1/3: Mengirim Market Order {entry_side} sebesar {amount} {market_info['base']}")
        entry_order = exchange.create_order(symbol, 'market', entry_side, amount)
        
        # Beri jeda singkat agar order masuk tereksekusi
        time.sleep(2)
        
        # 2. Kirim order Stop Loss (STOP_MARKET)
        print(f"2/3: Mengirim Stop Loss Order di harga {sl_price}")
        sl_params = {'stopPrice': sl_price, 'reduceOnly': True}
        exchange.create_order(symbol, 'STOP_MARKET', sl_tp_side, amount, params=sl_params)
        
        # 3. Kirim order Take Profit (TAKE_PROFIT_MARKET)
        print(f"3/3: Mengirim Take Profit Order di harga {tp_price}")
        tp_params = {'stopPrice': tp_price, 'reduceOnly': True}
        exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', sl_tp_side, amount, params=tp_params)

        print("✅ ✅ ✅ SEMUA ORDER BERHASIL DIBUAT UNTUK POSISI BARU ✅ ✅ ✅")
        print(f"Simbol: {symbol}, Sinyal: {signal}, Harga Masuk: ~{entry_order['price']}")

    except Exception as e:
        print(f"❌ ❌ ❌ GAGAL MEMBUAT ORDER LENGKAP untuk {symbol}: {e} ❌ ❌ ❌")

# ==============================================================================
# LOOP UTAMA BOT (VERSI MODIFIKASI - HANYA 1 POSISI TOTAL)
# ==============================================================================
if __name__ == '__main__':
    print("Bot Trading Dimulai... Mode: Memindai semua pasar, membuka maks 1 posisi.")
    print("---")
    
    while True:
        try:
            # 1. Cek apakah sudah ada posisi APAPUN yang terbuka
            open_positions = exchange.fetch_positions()
            active_position_symbols = {p['info']['symbol'] for p in open_positions if float(p['info']['positionAmt']) != 0}
            
            # 2. Jika TIDAK ada posisi terbuka, baru cari peluang
            if not active_position_symbols:
                print(f"[{time.ctime()}] Tidak ada posisi terbuka. Mencari peluang di seluruh pasar...")
                
                # 3. Jalankan scanner untuk mencari token potensial
                potential_symbols = scan_markets()
                
                if not potential_symbols:
                    print("Tidak ada token potensial yang ditemukan saat ini.")
                
                # 4. Analisis setiap token potensial, berhenti setelah menemukan satu peluang
                for symbol in potential_symbols:
                    print(f"Menganalisis sinyal untuk {symbol}...")
                    signal = get_signal(symbol, TIMEFRAME)
                    
                    if signal in ["LONG", "SHORT"]:
                        # Jika sinyal valid ditemukan, hitung SL/TP dan eksekusi
                        sl_price, tp_price = calculate_dynamic_sl_tp(symbol, TIMEFRAME, signal)
                        
                        if sl_price is not None and tp_price is not None:
                            execute_trade(symbol, signal, sl_price, tp_price)
                            
                            # 5. SANGAT PENTING: Hentikan loop setelah 1 trade berhasil dibuka
                            print(f"Posisi untuk {symbol} telah dibuka. Menghentikan pemindaian untuk siklus ini.")
                            break # <-- Ini akan menghentikan for loop
                    else:
                        print(f"Sinyal untuk {symbol}: {signal}. Tidak ada aksi.")
            
            # Jika SUDAH ADA posisi yang terbuka
            else:
                print(f"[{time.ctime()}] Posisi sudah terbuka untuk {active_position_symbols}. Bot akan menunggu hingga posisi ditutup.")

        except Exception as e:
            print(f"Terjadi error besar di loop utama: {e}")

        finally:
            # Jeda sebelum siklus berikutnya
            print(f"--- Siklus selesai. Menunggu {SCAN_INTERVAL_SECONDS} detik untuk pengecekan berikutnya. ---")
            time.sleep(SCAN_INTERVAL_SECONDS)

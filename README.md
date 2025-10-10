## ✅ `README.md`

```markdown
# Bot Trading AI - Winrate 80% (Simulasi)

> **BOT INI TIDAK MENJAMIN KEMENANGAN, BOT INI BERTUJUAN UNTUK EKSPERIMEN ATAU FOR FUN SAJA**

Bot ini dirancang untuk trading otomatis di **Binance Futures** dengan analisis teknikal lanjutan, **machine learning**, dan filter multi-indikator untuk meningkatkan akurasi sinyal hingga mendekati **80% winrate** (simulasi).

> ⚠️ Gunakan dengan risiko yang kamu pahami. Tidak ada jaminan PROFIT.

---

## ✅ Fitur Utama

- **Multi-Indikator**: EMA, RSI, MACD, Bollinger Bands, Stochastic RSI, ADX, ATR
- **Machine Learning**: RandomForest untuk prediksi arah harga
- **Analisis Sentimen**: (Dapat ditambahkan)
- **Manajemen Risiko**: SL/TP dinamis, leverage, dan position size
- **Backtesting Built-in**: Hitung winrate historis
- **Timeframe**: 15m (dapat diubah)

---

## 🛠️ Prasyarat

- Python 3.12 atau lebih tinggi
- Git
- Akun Binance Futures dengan API aktif

---

## 🔧 Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/willicc/tradebot-fun.git
cd tradebot-fun
```

### 2. Buat Virtual Environment (Opsional tapi disarankan)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Cara Mendapatkan API Binance

1. Buka [Binance Futures](https://www.binance.com/en/futures)
2. Login ke akun kamu
3. Masuk ke **Profil > API Management**
4. Klik **"Create API"**
5. Isi informasi dan centang:
   - ✅ Enable Futures
   - ✅ Enable Trading
6. Salin **API Key** dan **Secret Key**

> ⚠️ Simpan API Key dan Secret Key dengan aman. Jangan pernah membagikannya ke publik.

---

## ⚙️ Konfigurasi Bot

### 1. Edit File `main.py`

Buka file `main.py` dan ganti:

```python
BINANCE_API_KEY = 'GANTI_DENGAN_API_KEY_ANDA'
BINANCE_SECRET_KEY = 'GANTI_DENGAN_SECRET_KEY_ANDA'
```

### 2. Atur Parameter Strategi (Opsional)

Kamu bisa mengubah:
- `EMA_FAST_PERIOD`, `EMA_SLOW_PERIOD`
- `LEVERAGE`
- `POSITION_SIZE_USDT`
- `ATR_SL_MULTIPLIER`, `ATR_TP_MULTIPLIER`
- `ML_PREDICTION_THRESHOLD`

---

## ▶️ Menjalankan Bot

### 1. Jalankan Bot Secara Langsung

```bash
python main.py
```

Bot akan:
- Memindai pasar aktif
- Mencari peluang dengan volume tinggi
- Menghasilkan sinyal berdasarkan analisis multi-filter + ML
- Membuka posisi otomatis jika sinyal valid

---

## 🧪 Menjalankan Backtesting

Untuk menghitung winrate historis, kamu bisa menambahkan fungsi backtest di akhir `main.py`:

```python
# Contoh: Backtest untuk BTC/USDT selama 7 hari
winrate = run_backtest("BTC/USDT", "15m", days=7)
print(f"Winrate: {winrate}%")
```

---

## 📊 Winrate & Performa

- Bot ini dirancang untuk **winrate tinggi** (hingga 80%) dengan **risk-reward rendah** (1.5:1)
- Winrate tinggi tidak selalu berarti profit. Pastikan tetap gunakan manajemen risiko

---

## ⚠️ Peringatan

- Gunakan modal yang siap hilang
- Pastikan kamu mengerti risiko trading
- Bot ini tidak cocok untuk pemula
- Bot bisa berjalan 24/7, pantau terus performanya

---

## 🤝 Kontribusi

Jika kamu ingin berkontribusi atau menemukan bug, silakan buat issue atau pull request.

---

## 📄 Lisensi

Bot ini hanya untuk eksperimen dan edukasi. Tidak ada jaminan profit.

---

## 🎉 Semoga Menghibur & Sukses!

---

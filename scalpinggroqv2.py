import ccxt
import requests
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import json
import time
import os
from pathlib import Path
import re
import numpy as np

# =============================================================================
# LOAD GROQ API KEYS
# =============================================================================
def load_groq_keys():
    """Load Groq API keys dari api.txt"""
    try:
        if not os.path.exists('api.txt'):
            print("[ERROR] api.txt not found!")
            return []
        
        with open('api.txt', 'r') as f:
            keys = [line.strip() for line in f if line.strip()]
        
        print(f"[SUCCESS] Loaded {len(keys)} Groq API keys")
        return keys
    except Exception as e:
        print(f"[ERROR] Error loading API keys: {e}")
        return []

# =============================================================================
# BINANCE CONFIGURATION
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
# 1-MINUTE SCALPING CONFIGURATION
# =============================================================================
TIMEFRAME = '1m'                  # 1-minute untuk scalping
POSITION_SIZE_USDT = 50           # Kecil untuk scalping 1m
LEVERAGE = 2                      # Conservative untuk scalping
ATR_SL_MULTIPLIER = 0.5          # SANGAT TIGHT untuk 1m
ATR_TP_MULTIPLIER = 0.8          # Target 0.4-0.6% profit
MAX_POSITIONS = 5                 # Max concurrent
MIN_VOLUME_USDT = 1_000_000       # Liquid pairs only

# 1-MINUTE SPECIFIC INDICATORS
EMA_FAST = 3                      # Ultra-fast untuk 1m
EMA_MID = 7                       # Fast untuk 1m
EMA_SLOW = 14                     # Standard
RSI_PERIOD = 5                    # Very fast untuk 1m
RSI_OVERBOUGHT = 80               # Tighter untuk 1m
RSI_OVERSOLD = 20                 # Tighter untuk 1m
STOCH_K = 3                       # Very fast
STOCH_D = 3                       # Very fast

# Groq Configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_KEYS = load_groq_keys()
CURRENT_KEY_INDEX = 0

# =============================================================================
# LOGGER
# =============================================================================
class Logger:
    OKGREEN = '\033[92m'
    OKCYAN = '\033[96m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'
    
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
    def ai(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {Logger.BOLD}[🤖 AI]{Logger.ENDC} {msg}")
    
    @staticmethod
    def trade(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {Logger.BOLD}[💹 TRADE]{Logger.ENDC} {msg}")

# =============================================================================
# GROQ API CALLER
# =============================================================================
class GroqAnalyzer:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.current_index = 0
        self.max_retries = 3
    
    def get_current_key(self):
        if not self.api_keys:
            raise Exception("No API keys available!")
        return self.api_keys[self.current_index % len(self.api_keys)]
    
    def switch_to_next_key(self):
        self.current_index += 1
        Logger.warning(f"Switching to API key #{self.current_index % len(self.api_keys) + 1}")
    
    def call_groq(self, prompt, model="mixtral-8x7b-32768", temperature=0.3):
        """Call Groq API dengan fallback"""
        for attempt in range(self.max_retries):
            try:
                api_key = self.get_current_key()
                
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert 1-minute scalping analyst. Focus on HIGH PROBABILITY trades with 90%+ accuracy. Provide EXACT prices, not ranges. Be concise and actionable."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": 1500
                }
                
                response = requests.post(
                    GROQ_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                
                elif response.status_code == 429:  # Rate limit
                    Logger.warning(f"Rate limited on key #{self.current_index % len(self.api_keys) + 1}")
                    self.switch_to_next_key()
                    time.sleep(2)
                
                elif response.status_code == 401:  # Invalid API key
                    Logger.error(f"Invalid API key #{self.current_index % len(self.api_keys) + 1}")
                    self.switch_to_next_key()
                
                else:
                    Logger.error(f"API Error: {response.status_code}")
                    self.switch_to_next_key()
            
            except requests.exceptions.Timeout:
                Logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}")
                time.sleep(1)
            
            except Exception as e:
                Logger.error(f"Error: {e}")
                self.switch_to_next_key()
        
        return None

# =============================================================================
# 1-MINUTE SCALPING PROMPTS (OPTIMIZED)
# =============================================================================
class ScalpingPrompts:
    @staticmethod
    def micro_trend_scanner():
        """Prompt 1: Micro-trend scanning untuk 1m"""
        return """
        ANALYZE FOR 1-MINUTE SCALPING OPPORTUNITIES:
        
        Scan crypto pairs (BTC, ETH, SOL, XRP, DOGE) dan identify EXACTLY 5 IMMEDIATE trading opportunities.
        
        For EACH opportunity, provide in EXACT format:
        
        SCALP 1:
        - Symbol: [EXACT SYMBOL]
        - Direction: [LONG/SHORT - MUST CHOOSE]
        - Entry: [EXACT PRICE - 8 DECIMALS]
        - TP1: [EXACT PRICE]
        - TP2: [EXACT PRICE]
        - SL: [EXACT PRICE]
        - Confidence: [90-99%]
        - Setup: [ONE LINE - what makes it 90%+ likely]
        
        CRITICAL FOR 90% WINRATE:
        - Only identify HIGH PROBABILITY setups
        - Entry MUST be on strong 1m candle confirmation
        - SL must be TIGHT (0.3-0.5% below entry)
        - TP must be REALISTIC (0.5-1% from entry)
        - Look for EMA 3/7 crossover + RSI 5 confirmation
        - Avoid wide spreads, only liquid pairs
        
        Provide 5 IMMEDIATE scalps. NO analysis text, ONLY trades.
        """
    
    @staticmethod
    def momentum_scalper():
        """Prompt 2: Momentum-based scalping"""
        return """
        1-MINUTE MOMENTUM SCALPING:
        
        Identify 5 coins in STRONG momentum on 1m chart. For each, provide:
        
        - Symbol: [SYMBOL]
        - Direction: [LONG/SHORT]
        - Entry: [EXACT PRICE]
        - TP: [EXACT PRICE]
        - SL: [EXACT PRICE]
        - Reason: [Why momentum will continue for 3-5 candles]
        - Win%: [Your confidence 90-99%]
        
        Key for 90% winrate:
        - Only trade when RSI 5 is 30-70 (not at extremes)
        - EMA3 > EMA7 for LONG, EMA3 < EMA7 for SHORT
        - Volume must be ABOVE average
        - Setup must have MULTIPLE confirmations
        """
    
    @staticmethod
    def breakout_scalper():
        """Prompt 3: Breakout scalping 1m"""
        return """
        1-MINUTE BREAKOUT SCALPING:
        
        Find symbols breaking out of tight 1m consolidation.
        
        For 5 breakout opportunities provide:
        
        - Symbol: [SYMBOL]
        - Direction: [BREAKOUT UP/DOWN]
        - Consolidation Range: [HIGH-LOW]
        - Break Point: [EXACT PRICE]
        - Entry: [EXACT PRICE]
        - Target: [EXACT PRICE - usually 1.5x breakout distance]
        - SL: [EXACT PRICE - consolidation opposite side]
        - Setup Quality: [What confirms real breakout, not fake]
        
        For 90%+ accuracy:
        - Volume must INCREASE at breakout
        - False breakouts common in 1m, need confirmation
        - Risk/Reward must be at least 1:1.5
        """
    
    @staticmethod
    def mean_reversion_scalper():
        """Prompt 4: Mean reversion 1m"""
        return """
        1-MINUTE MEAN REVERSION SCALPING:
        
        Find symbols at extremes (RSI < 20 or > 80) that will revert.
        
        5 Reversion opportunities:
        
        - Symbol: [SYMBOL]
        - Current RSI5: [VALUE]
        - Entry: [EXACT PRICE]
        - Expected Reversion: [TARGET PRICE]
        - SL: [Extreme level]
        - TP: [Middle of range usually]
        - Confidence: [95-99% for oversold/overbought]
        
        Key insights:
        - RSI5 extreme usually reverts within 2-4 candles
        - Use BB (upper/lower) for target
        - MUST have volume confirmation
        - Best when trend is strong (less whipsaw)
        """
    
    @staticmethod
    def grid_scalper():
        """Prompt 5: Grid trading scalper"""
        return """
        1-MINUTE GRID SCALPING:
        
        Identify symbols in RANGE (not trending). Set grid of small profitable scalps.
        
        5 Grid opportunities:
        
        - Symbol: [SYMBOL]
        - Range High: [EXACT PRICE]
        - Range Low: [EXACT PRICE]
        - First Entry: [EXACT PRICE]
        - Grid Size: [EXACT PRICE PER GRID]
        - Each Grid TP: [EXACT PRICE - small %]
        - Grid Layers: [How many grids to set]
        - Confidence: [90%+ range will hold]
        
        For 90% winrate in range:
        - Range must be CONFIRMED (tested 2+ times)
        - Support/resistance must be STRONG
        - Grid size = 0.3-0.5% of price
        - Perfect for sideways 1m action
        """
    
    @staticmethod
    def multi_timeframe_scalper():
        """Prompt 6: Multi-timeframe confirmation"""
        return """
        1-MINUTE SCALPING WITH 5-MINUTE CONFIRMATION:
        
        Use 5m trend as bias, trade 1m entries ONLY when aligned.
        
        5 High-probability entries:
        
        - Symbol: [SYMBOL]
        - 5m Trend: [UP/DOWN/NEUTRAL]
        - 1m Entry Signal: [EXACT SETUP]
        - Entry Price: [EXACT]
        - TP: [EXACT]
        - SL: [EXACT]
        - Why Aligned: [How 1m entry + 5m trend = high probability]
        - Win Probability: [95-99%]
        
        This filter ELIMINATES counter-trend trades = higher winrate.
        Only trade 1m entries when 5m trend SUPPORTS direction.
        """
    
    @staticmethod
    def vwap_scalper():
        """Prompt 7: VWAP-based scalping"""
        return """
        1-MINUTE VWAP SCALPING:
        
        Trade bounces off VWAP (volume-weighted avg price).
        
        5 VWAP scalp opportunities:
        
        - Symbol: [SYMBOL]
        - Current Price: [EXACT]
        - VWAP Level: [EXACT]
        - Distance to VWAP: [%]
        - Entry (bounce from VWAP): [EXACT]
        - TP (target after bounce): [EXACT]
        - SL: [Opposite side of VWAP]
        - Bounce Probability: [96-99% - VWAP is strong level]
        
        VWAP bounces are HIGH PROBABILITY on 1m.
        Mean reversion to VWAP = 95%+ accurate.
        """

# =============================================================================
# TRADE PARSER UNTUK 1M DATA
# =============================================================================
class TradeParser:
    @staticmethod
    def parse_1m_trades(response):
        """Parse trades dari Groq response untuk 1m data"""
        trades = []
        
        try:
            # Split by SCALP X:
            trade_blocks = re.split(r'SCALP\s+\d+:|TRADE\s+\d+:', response)
            
            for block in trade_blocks[1:]:
                trade = {}
                
                # Parse Symbol
                symbol_match = re.search(r'Symbol:\s*\[?(\w+)\]?', block)
                if symbol_match:
                    sym = symbol_match.group(1)
                    # Ensure /USDT suffix
                    if '/' not in sym:
                        sym = f"{sym}/USDT"
                    trade['symbol'] = sym
                
                # Parse Direction
                direction_match = re.search(r'Direction:\s*\[?(LONG|SHORT|UP|DOWN)\]?', block)
                if direction_match:
                    direction = direction_match.group(1)
                    trade['direction'] = 'LONG' if direction in ['LONG', 'UP'] else 'SHORT'
                
                # Parse Entry Price (EXACT required)
                entry_match = re.search(r'Entry(?:\s+Price)?:\s*\[?([0-9.]+)\]?', block)
                if entry_match:
                    trade['entry'] = float(entry_match.group(1))
                
                # Parse TP1 or Target
                tp_match = re.search(r'TP(?:1)?:\s*\[?([0-9.]+)\]?|Target:\s*\[?([0-9.]+)\]?', block)
                if tp_match:
                    tp_val = tp_match.group(1) or tp_match.group(2)
                    trade['tp'] = float(tp_val)
                
                # Parse TP2 (optional)
                tp2_match = re.search(r'TP2:\s*\[?([0-9.]+)\]?', block)
                if tp2_match:
                    trade['tp2'] = float(tp2_match.group(1))
                
                # Parse Stop Loss (CRITICAL untuk 1m)
                sl_match = re.search(r'SL:\s*\[?([0-9.]+)\]?|Stop Loss:\s*\[?([0-9.]+)\]?', block)
                if sl_match:
                    sl_val = sl_match.group(1) or sl_match.group(2)
                    trade['sl'] = float(sl_val)
                
                # Parse Confidence
                conf_match = re.search(r'(?:Confidence|Win%|Win\%):\s*\[?(\d+)%?\]?', block)
                if conf_match:
                    trade['confidence'] = int(conf_match.group(1))
                else:
                    trade['confidence'] = 85  # Default
                
                # Parse Reason/Setup
                reason_match = re.search(r'(?:Setup|Reason|Setup Quality):\s*\[?([^\]]+)\]?', block)
                if reason_match:
                    trade['reason'] = reason_match.group(1).strip()[:100]
                
                # Validate 1m scalp requirements
                if trade and 'symbol' in trade and 'entry' in trade and 'tp' in trade and 'sl' in trade:
                    # Validate tight SL for 1m
                    sl_distance = abs(trade['entry'] - trade['sl']) / trade['entry'] * 100
                    tp_distance = abs(trade['tp'] - trade['entry']) / trade['entry'] * 100
                    
                    # For 1m scalping: SL should be 0.3-0.8%, TP 0.5-1.2%
                    if 0.2 < sl_distance < 2.0 and 0.3 < tp_distance < 3.0:
                        trade['sl_pct'] = sl_distance
                        trade['tp_pct'] = tp_distance
                        trade['rr_ratio'] = tp_distance / sl_distance
                        trades.append(trade)
        
        except Exception as e:
            Logger.error(f"Error parsing trades: {e}")
        
        return trades

# =============================================================================
# TRADE EXECUTOR UNTUK 1M
# =============================================================================
class OneMinuteTradeExecutor:
    def __init__(self, exchange, position_size=50, leverage=2):
        self.exchange = exchange
        self.position_size = position_size
        self.leverage = leverage
        self.active_positions = {}
    
    def validate_scalp_trade(self, trade):
        """Validate 1m scalp trade parameters"""
        required = ['symbol', 'direction', 'entry', 'tp', 'sl']
        if not all(field in trade for field in required):
            return False, "Missing required fields"
        
        # Check tight SL requirement
        sl_pct = abs(trade['entry'] - trade['sl']) / trade['entry'] * 100
        if sl_pct < 0.2 or sl_pct > 2.0:
            return False, f"SL {sl_pct:.3f}% outside 1m range"
        
        # Check TP requirement
        tp_pct = abs(trade['tp'] - trade['entry']) / trade['entry'] * 100
        if tp_pct < 0.3 or tp_pct > 3.0:
            return False, f"TP {tp_pct:.3f}% outside 1m range"
        
        # Check R:R ratio for 1m (should be 1:1.5+)
        rr = tp_pct / sl_pct
        if rr < 1.2:
            return False, f"R:R ratio {rr:.2f} too low"
        
        return True, "Valid"
    
    def execute_1m_scalp(self, trade):
        """Execute 1m scalp trade"""
        try:
            valid, msg = self.validate_scalp_trade(trade)
            if not valid:
                Logger.warning(f"Invalid scalp: {msg}")
                return False
            
            symbol = trade['symbol']
            direction = trade['direction']
            entry = trade['entry']
            tp = trade['tp']
            sl = trade['sl']
            
            Logger.trade(f"🔥 1m SCALP: {direction} {symbol} @ {entry:.8f}")
            Logger.trade(f"TP: {tp:.8f} | SL: {sl:.8f} | RR: {trade.get('rr_ratio', 0):.2f}:1")
            
            # Set leverage
            self.exchange.set_leverage(self.leverage, symbol)
            time.sleep(0.3)
            
            # Calculate position
            amount = self.position_size / entry
            amount = float(self.exchange.amount_to_precision(symbol, amount))
            
            # Market order
            side = 'buy' if direction == "LONG" else 'sell'
            tp_side = 'sell' if direction == "LONG" else 'buy'
            
            Logger.info(f"Executing {side} {amount} {symbol}")
            order = self.exchange.create_order(symbol, 'market', side, amount)
            Logger.success(f"Market order: {order['id']}")
            
            time.sleep(0.8)
            
            # SL order (CRITICAL untuk 1m)
            sl_price = float(self.exchange.price_to_precision(symbol, sl))
            sl_params = {'stopPrice': sl_price, 'reduceOnly': True}
            sl_order = self.exchange.create_order(symbol, 'STOP_MARKET', tp_side, amount, params=sl_params)
            Logger.success(f"SL @ {sl_price}: {sl_order['id']}")
            
            time.sleep(0.4)
            
            # TP order
            tp_price = float(self.exchange.price_to_precision(symbol, tp))
            tp_params = {'stopPrice': tp_price, 'reduceOnly': True}
            tp_order = self.exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', tp_side, amount, params=tp_params)
            Logger.success(f"TP @ {tp_price}: {tp_order['id']}")
            
            self.active_positions[symbol] = {
                'direction': direction,
                'entry': entry,
                'sl': sl_price,
                'tp': tp_price,
                'amount': amount,
                'opened_at': datetime.now(),
                'reason': trade.get('reason', '1m Scalp'),
                'confidence': trade.get('confidence', 85)
            }
            
            Logger.success(f"✅ 1m SCALP {direction} {symbol} LIVE!")
            return True
        
        except Exception as e:
            Logger.error(f"Execution error: {e}")
            return False

# =============================================================================
# MAIN AI SCALPING BOT
# =============================================================================
class AIScalpingBot:
    def __init__(self, groq_keys, exchange, max_positions=5):
        self.analyzer = GroqAnalyzer(groq_keys)
        self.executor = OneMinuteTradeExecutor(exchange, POSITION_SIZE_USDT, LEVERAGE)
        self.max_positions = max_positions
        self.session = {
            'scalps_opened': 0,
            'scalps_closed': 0,
            'daily_pnl': 0,
            'analysis_runs': 0,
            'total_winrate': 0
        }
    
    def run_scalp_analysis(self, prompt_type=1):
        """Run 1-minute scalp analysis dengan Groq"""
        try:
            prompt_map = {
                1: ScalpingPrompts.micro_trend_scanner,
                2: ScalpingPrompts.momentum_scalper,
                3: ScalpingPrompts.breakout_scalper,
                4: ScalpingPrompts.mean_reversion_scalper,
                5: ScalpingPrompts.grid_scalper,
                6: ScalpingPrompts.multi_timeframe_scalper,
                7: ScalpingPrompts.vwap_scalper,
            }
            
            prompt_func = prompt_map.get(prompt_type, ScalpingPrompts.micro_trend_scanner)
            prompt = prompt_func()
            
            Logger.ai(f"Running 1m scalp analysis #{prompt_type}")
            response = self.analyzer.call_groq(prompt, temperature=0.2)
            
            if response:
                Logger.success(f"Analysis complete: {len(response)} chars")
                return response
            else:
                Logger.error("No response from Groq")
                return None
        
        except Exception as e:
            Logger.error(f"Analysis error: {e}")
            return None
    
    def execute_scalp_trades(self, analysis_response):
        """Execute trades dari AI analysis"""
        try:
            trades = TradeParser.parse_1m_trades(analysis_response)
            Logger.ai(f"Parsed {len(trades)} scalp opportunities")
            
            # Get current positions
            open_positions = self.executor.exchange.fetch_positions()
            active_symbols = {
                p['info']['symbol'] for p in open_positions
                if float(p['info']['positionAmt']) != 0
            }
            
            # Execute scalps
            for trade in trades:
                if len(active_symbols) >= self.max_positions:
                    Logger.warning(f"Max scalps ({self.max_positions}) reached")
                    break
                
                if trade['symbol'] in active_symbols:
                    Logger.warning(f"{trade['symbol']} already open")
                    continue
                
                if self.executor.execute_1m_scalp(trade):
                    active_symbols.add(trade['symbol'])
                    self.session['scalps_opened'] += 1
                    time.sleep(0.8)
        
        except Exception as e:
            Logger.error(f"Execution error: {e}")
    
    def run_scalp_cycle(self):
        """Run one scalp analysis cycle"""
        try:
            Logger.info("="*80)
            Logger.info(f"1m SCALP CYCLE: {datetime.now().strftime('%H:%M:%S')}")
            
            # Rotate through 7 different scalp strategies
            prompt_type = (self.session['analysis_runs'] % 7) + 1
            
            response = self.run_scalp_analysis(prompt_type)
            
            if response:
                Logger.ai("Analysis Response:")
                print(response[:400] + "..." if len(response) > 400 else response)
                
                self.execute_scalp_trades(response)
                self.session['analysis_runs'] += 1
        
        except Exception as e:
            Logger.error(f"Cycle error: {e}")
    
    def print_summary(self):
        """Print session summary"""
        print("\n" + "="*80)
        print("📊 1M SCALPING SESSION")
        print("="*80)
        print(f"Scalps Opened: {self.session['scalps_opened']}")
        print(f"Active Positions: {len(self.executor.active_positions)}")
        print(f"Analysis Runs: {self.session['analysis_runs']}")
        print(f"Daily PnL: ${self.session['daily_pnl']:.2f}")
        print("="*80 + "\n")
    
    def run_continuous(self, interval=60):
        """Run continuous 1m scalping"""
        Logger.success("🤖🔥 AI 1-MINUTE SCALPING BOT STARTED!")
        Logger.info(f"Target: 90%+ Winrate | Interval: {interval}s")
        Logger.info(f"Position Size: ${POSITION_SIZE_USDT} | Leverage: {LEVERAGE}x")
        
        try:
            while True:
                self.run_scalp_cycle()
                self.print_summary()
                
                Logger.info(f"⏳ Next 1m scalp scan in {interval}s...")
                time.sleep(interval)
        
        except KeyboardInterrupt:
            Logger.warning("Bot stopped by user")
            self.print_summary()

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("\n" + "="*80)
    print("🤖🔥 ADVANCED 1-MINUTE SCALPING BOT v5.0".center(80))
    print("Groq AI + 90% Winrate Target".center(80))
    print("="*80 + "\n")
    
    if not GROQ_KEYS:
        Logger.error("No Groq API keys in api.txt!")
        exit(1)
    
    bot = AIScalpingBot(GROQ_KEYS, exchange, MAX_POSITIONS)
    bot.run_continuous(interval=60)  # Every 1 minute

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

# =============================================================================
# LOAD GROQ API KEYS
# =============================================================================
def load_groq_keys():
    """Load Groq API keys dari api.txt"""
    try:
        if not os.path.exists('api.txt'):
            print("[ERROR] api.txt not found! Create file with Groq API keys (one per line)")
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
BINANCE_API_KEY = 'YOUR_API_KEY'
BINANCE_SECRET_KEY = 'YOUR_SECRET_KEY'

exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET_KEY,
    'options': {
        'defaultType': 'future',
        'enableRateLimit': True,
    },
})

# Trading Configuration
TIMEFRAME = '1m'
POSITION_SIZE_USDT = 100
LEVERAGE = 3
ATR_SL_MULTIPLIER = 0.7
ATR_TP_MULTIPLIER = 1.2
MAX_POSITIONS = 5

# Groq API Configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_KEYS = load_groq_keys()
CURRENT_KEY_INDEX = 0

# =============================================================================
# COLOR LOGGING
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

# =============================================================================
# GROQ API CALLER dengan Fallback
# =============================================================================
class GroqAnalyzer:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.current_index = 0
        self.max_retries = 3
    
    def get_current_key(self):
        """Get current API key, fallback to next if error"""
        if not self.api_keys:
            raise Exception("No API keys available!")
        return self.api_keys[self.current_index % len(self.api_keys)]
    
    def switch_to_next_key(self):
        """Switch to next API key"""
        self.current_index += 1
        Logger.warning(f"Switching to API key #{self.current_index % len(self.api_keys) + 1}")
    
    def call_groq(self, prompt, model="mixtral-8x7b-32768", temperature=0.7):
        """Call Groq API dengan fallback management"""
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
                        {"role": "system", "content": "You are an expert cryptocurrency and stock trading analyst. Provide clear, actionable trading recommendations."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": 2000
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
                    Logger.error(f"API Error: {response.status_code} - {response.text[:200]}")
                    self.switch_to_next_key()
            
            except requests.exceptions.Timeout:
                Logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}")
                time.sleep(1)
            
            except Exception as e:
                Logger.error(f"Error calling Groq: {e}")
                self.switch_to_next_key()
        
        return None

# =============================================================================
# TRADING PROMPTS LIBRARY
# =============================================================================
class TradingPrompts:
    @staticmethod
    def high_probability_trades(market_context="crypto"):
        """Prompt 1: High probability trades"""
        return f"""
        Analyze the current {market_context} market and identify EXACTLY FIVE high-probability trades.
        
        For EACH trade, provide in this EXACT format:
        
        TRADE 1:
        - Symbol: [SYMBOL]
        - Direction: [LONG/SHORT]
        - Entry Price: [PRICE]
        - Target 1: [PRICE]
        - Target 2: [PRICE]
        - Stop Loss: [PRICE]
        - Risk-Reward Ratio: [RATIO]
        - Confidence: [%]
        - Technical Reason: [ONE LINE]
        
        TRADE 2: ... (repeat for 5 trades)
        
        Focus on high-probability setups with clear technical reasons.
        """
    
    @staticmethod
    def technical_analysis(symbol):
        """Prompt 2: Technical analysis"""
        return f"""
        Analyze {symbol} using DAILY and WEEKLY timeframes.
        
        Provide:
        1. Key Support Levels: [List 3 levels]
        2. Key Resistance Levels: [List 3 levels]
        3. Trendline Analysis: [Current trend]
        4. Moving Averages (20, 50, 200): [Positioning]
        5. Momentum Indicators (RSI, MACD): [Status]
        
        Final Signal: [BUY / HOLD / SELL]
        Confidence: [%]
        
        Provide step-by-step reasoning for your signal.
        """
    
    @staticmethod
    def news_to_trade(symbol_or_sector):
        """Prompt 3: News-to-trade converter"""
        return f"""
        Summarize the most recent news about {symbol_or_sector} and convert to trading insights.
        
        Provide:
        1. Key News Summary: [BRIEF]
        2. Short-term Impact: [BULLISH/BEARISH/NEUTRAL]
        3. Long-term Impact: [BULLISH/BEARISH/NEUTRAL]
        4. Expected Price Movement: [RANGE]
        5. Suggested Position: [LONG/SHORT/NONE]
        6. Entry Price Suggestion: [PRICE]
        7. Target & Stop-Loss: [LEVELS]
        
        Base only on verifiable recent news, not speculation.
        """
    
    @staticmethod
    def strategy_backtest(strategy, symbol, period="3 months"):
        """Prompt 4: Strategy backtester"""
        return f"""
        Backtest the {strategy} strategy on {symbol} over the past {period}.
        
        Provide:
        1. Win Rate: [%]
        2. Profit Factor: [RATIO]
        3. Maximum Drawdown: [%]
        4. Average Win: [%]
        5. Average Loss: [%]
        6. Total Return: [%]
        
        Suggest 3 improvements to enhance performance.
        """
    
    @staticmethod
    def portfolio_risk_manager(portfolio_str):
        """Prompt 5: Portfolio risk manager"""
        return f"""
        Evaluate this portfolio: {portfolio_str}
        
        Analyze:
        1. Overexposed Areas: [SECTORS]
        2. Weak Positions: [SYMBOLS]
        3. Correlation Issues: [PROBLEMS]
        4. Recommended Rebalancing: [ACTIONS]
        5. Hedging Strategies: [FOR 20% DECLINE]
        
        Provide specific rebalancing percentages.
        """
    
    @staticmethod
    def trading_journal_analyzer(trades_str):
        """Prompt 6: Trading journal analyzer"""
        return f"""
        Analyze these 20 recent trades:
        {trades_str}
        
        Identify:
        1. Recurring Errors: [LIST]
        2. Missed Opportunities: [LIST]
        3. Behavioral Biases: [LIST]
        
        Provide 3 personalized rules to improve immediately.
        """
    
    @staticmethod
    def daily_trading_plan(market_or_asset):
        """Prompt 7: Daily trading plan"""
        return f"""
        Create a structured daily trading plan for {market_or_asset}.
        
        Include:
        1. PRE-MARKET (08:00): [SCAN & SETUP]
        2. OPEN (09:30): [EXECUTION STRATEGY]
        3. MIDDAY (12:00): [ADJUSTMENTS]
        4. AFTERNOON (15:00): [POSITION MANAGEMENT]
        5. CLOSE (16:00): [CLOSING APPROACH]
        
        Format as time-stamped checklist.
        """

# =============================================================================
# TRADE PARSER dari AI Response
# =============================================================================
class TradeParser:
    @staticmethod
    def parse_trades_from_response(response):
        """Parse trades dari Groq response"""
        trades = []
        
        try:
            # Split by TRADE X:
            trade_blocks = re.split(r'TRADE\s+\d+:', response)
            
            for block in trade_blocks[1:]:  # Skip first empty split
                trade = {}
                
                # Parse Symbol
                symbol_match = re.search(r'Symbol:\s*\[?(\w+)\]?', block)
                if symbol_match:
                    trade['symbol'] = symbol_match.group(1)
                
                # Parse Direction
                direction_match = re.search(r'Direction:\s*\[?(LONG|SHORT)\]?', block)
                if direction_match:
                    trade['direction'] = direction_match.group(1)
                
                # Parse Entry Price
                entry_match = re.search(r'Entry Price:\s*\[?([0-9.]+)\]?', block)
                if entry_match:
                    trade['entry'] = float(entry_match.group(1))
                
                # Parse Stop Loss
                sl_match = re.search(r'Stop Loss:\s*\[?([0-9.]+)\]?', block)
                if sl_match:
                    trade['stop_loss'] = float(sl_match.group(1))
                
                # Parse Target 1
                target1_match = re.search(r'Target 1:\s*\[?([0-9.]+)\]?', block)
                if target1_match:
                    trade['target1'] = float(target1_match.group(1))
                
                # Parse Target 2
                target2_match = re.search(r'Target 2:\s*\[?([0-9.]+)\]?', block)
                if target2_match:
                    trade['target2'] = float(target2_match.group(1))
                
                # Parse Risk-Reward Ratio
                rr_match = re.search(r'Risk-Reward Ratio:\s*\[?([0-9.:]+)\]?', block)
                if rr_match:
                    trade['risk_reward'] = rr_match.group(1)
                
                # Parse Confidence
                conf_match = re.search(r'Confidence:\s*\[?(\d+)%?\]?', block)
                if conf_match:
                    trade['confidence'] = int(conf_match.group(1))
                
                # Parse Reason
                reason_match = re.search(r'Technical Reason:\s*\[?([^\]]+)\]?', block)
                if reason_match:
                    trade['reason'] = reason_match.group(1).strip()
                
                if trade and 'symbol' in trade:
                    trades.append(trade)
        
        except Exception as e:
            Logger.error(f"Error parsing trades: {e}")
        
        return trades
    
    @staticmethod
    def parse_signal_from_response(response):
        """Parse BUY/SELL/HOLD signal"""
        signal_match = re.search(r'(BUY|SELL|HOLD)', response, re.IGNORECASE)
        if signal_match:
            return signal_match.group(1).upper()
        return "HOLD"
    
    @staticmethod
    def parse_confidence_from_response(response):
        """Parse confidence percentage"""
        conf_match = re.search(r'Confidence:\s*(\d+)%?', response)
        if conf_match:
            return int(conf_match.group(1))
        return 0

# =============================================================================
# TRADE EXECUTOR
# =============================================================================
class TradeExecutor:
    def __init__(self, exchange, position_size=100, leverage=3):
        self.exchange = exchange
        self.position_size = position_size
        self.leverage = leverage
        self.active_positions = {}
    
    def validate_trade(self, trade):
        """Validate trade parameters"""
        required_fields = ['symbol', 'direction', 'entry', 'stop_loss', 'target1']
        return all(field in trade for field in required_fields)
    
    def format_symbol(self, symbol):
        """Ensure symbol has /USDT suffix"""
        if '/' not in symbol:
            symbol = f"{symbol}/USDT"
        return symbol
    
    def calculate_position_size(self, entry_price):
        """Calculate position size based on capital"""
        return self.position_size / entry_price
    
    def calculate_sl_tp_from_trade(self, trade):
        """Use AI-provided SL/TP"""
        return trade['stop_loss'], trade['target1']
    
    async def execute_trade(self, trade):
        """Execute trade on Binance"""
        try:
            if not self.validate_trade(trade):
                Logger.error(f"Invalid trade: missing required fields")
                return False
            
            symbol = self.format_symbol(trade['symbol'])
            direction = trade['direction']
            entry = trade['entry']
            
            Logger.info(f"Executing {direction} trade for {symbol}")
            Logger.info(f"Entry: ${entry:.8f} | Target: ${trade['target1']:.8f} | SL: ${trade['stop_loss']:.8f}")
            
            # Set leverage
            self.exchange.set_leverage(self.leverage, symbol)
            time.sleep(0.5)
            
            # Calculate position
            amount = self.calculate_position_size(entry)
            amount = float(self.exchange.amount_to_precision(symbol, amount))
            
            # Market order
            side = 'buy' if direction == "LONG" else 'sell'
            tp_side = 'sell' if direction == "LONG" else 'buy'
            
            Logger.info(f"Executing market order: {side} {amount} {symbol}")
            order = self.exchange.create_order(symbol, 'market', side, amount)
            Logger.success(f"Market order executed: {order['id']}")
            
            time.sleep(1.5)
            
            # Set SL
            sl_price = float(self.exchange.price_to_precision(symbol, trade['stop_loss']))
            sl_params = {'stopPrice': sl_price, 'reduceOnly': True}
            sl_order = self.exchange.create_order(symbol, 'STOP_MARKET', tp_side, amount, params=sl_params)
            Logger.success(f"SL order set: {sl_order['id']}")
            
            time.sleep(0.5)
            
            # Set TP
            tp_price = float(self.exchange.price_to_precision(symbol, trade['target1']))
            tp_params = {'stopPrice': tp_price, 'reduceOnly': True}
            tp_order = self.exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', tp_side, amount, params=tp_params)
            Logger.success(f"TP order set: {tp_order['id']}")
            
            self.active_positions[symbol] = {
                'direction': direction,
                'entry': entry,
                'sl': sl_price,
                'tp': tp_price,
                'amount': amount,
                'opened_at': datetime.now(),
                'reason': trade.get('reason', 'AI Analysis')
            }
            
            Logger.success(f"✅ Position {direction} for {symbol} fully configured!")
            return True
        
        except Exception as e:
            Logger.error(f"Error executing trade: {e}")
            return False

# =============================================================================
# MAIN BOT CLASS
# =============================================================================
class AITradingBot:
    def __init__(self, groq_keys, exchange, max_positions=5):
        self.analyzer = GroqAnalyzer(groq_keys)
        self.executor = TradeExecutor(exchange, POSITION_SIZE_USDT, LEVERAGE)
        self.max_positions = max_positions
        self.session_data = {
            'trades_opened': 0,
            'trades_closed': 0,
            'daily_pnl': 0,
            'analysis_count': 0
        }
    
    def analyze_market_with_groq(self, prompt_type="high_probability"):
        """Analyze market menggunakan Groq AI"""
        try:
            Logger.ai(f"Running analysis: {prompt_type}")
            
            # Get prompt
            if prompt_type == "high_probability":
                prompt = TradingPrompts.high_probability_trades("crypto")
            elif prompt_type == "technical":
                prompt = TradingPrompts.technical_analysis("BTC/USDT")
            elif prompt_type == "news":
                prompt = TradingPrompts.news_to_trade("Bitcoin")
            elif prompt_type == "backtest":
                prompt = TradingPrompts.strategy_backtest("moving average crossover", "BTC/USDT", "1 month")
            elif prompt_type == "daily_plan":
                prompt = TradingPrompts.daily_trading_plan("crypto futures")
            else:
                prompt = TradingPrompts.high_probability_trades()
            
            # Call Groq
            response = self.analyzer.call_groq(prompt)
            
            if response:
                Logger.success(f"Analysis complete: {len(response)} chars")
                return response
            else:
                Logger.error("No response from Groq")
                return None
        
        except Exception as e:
            Logger.error(f"Analysis error: {e}")
            return None
    
    def execute_ai_trades(self, analysis_response):
        """Execute trades berdasarkan AI analysis"""
        try:
            # Parse trades dari response
            trades = TradeParser.parse_trades_from_response(analysis_response)
            Logger.ai(f"Parsed {len(trades)} trade recommendations")
            
            # Get current positions
            open_positions = self.executor.exchange.fetch_positions()
            active_symbols = {
                p['info']['symbol'] for p in open_positions
                if float(p['info']['positionAmt']) != 0
            }
            
            # Execute trades
            for trade in trades:
                if len(active_symbols) >= self.max_positions:
                    Logger.warning(f"Max positions reached ({self.max_positions})")
                    break
                
                symbol = TradeExecutor.format_symbol(None, trade['symbol'])
                if symbol in active_symbols:
                    Logger.warning(f"{symbol} already has open position")
                    continue
                
                # Execute
                if self.executor.execute_trade(trade):
                    active_symbols.add(symbol)
                    self.session_data['trades_opened'] += 1
                    time.sleep(1)
        
        except Exception as e:
            Logger.error(f"Error executing trades: {e}")
    
    def run_analysis_cycle(self, prompt_type="high_probability"):
        """Run satu analysis cycle"""
        try:
            Logger.info("="*80)
            Logger.info(f"Analysis Cycle: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Analyze
            response = self.analyze_market_with_groq(prompt_type)
            
            if response:
                Logger.ai("Analysis Response Preview:")
                print(response[:500] + "..." if len(response) > 500 else response)
                
                # Execute trades
                self.execute_ai_trades(response)
                self.session_data['analysis_count'] += 1
        
        except Exception as e:
            Logger.error(f"Cycle error: {e}")
    
    def print_session_summary(self):
        """Print session summary"""
        print("\n" + "="*80)
        print("📊 SESSION SUMMARY")
        print("="*80)
        print(f"Analysis Runs: {self.session_data['analysis_count']}")
        print(f"Trades Opened: {self.session_data['trades_opened']}")
        print(f"Active Positions: {len(self.executor.active_positions)}")
        print("="*80 + "\n")
    
    def run_continuous(self, interval=300):
        """Run bot continuously"""
        prompts = [
            "high_probability",
            "technical",
            "news",
            "backtest",
            "daily_plan"
        ]
        prompt_index = 0
        
        Logger.success("🤖 AI Trading Bot Started!")
        Logger.info(f"Analyzing with Groq AI every {interval}s")
        
        try:
            while True:
                prompt = prompts[prompt_index % len(prompts)]
                self.run_analysis_cycle(prompt)
                prompt_index += 1
                
                self.print_session_summary()
                
                Logger.info(f"Waiting {interval}s until next analysis...")
                time.sleep(interval)
        
        except KeyboardInterrupt:
            Logger.warning("Bot stopped by user")
            self.print_session_summary()

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    print("\n" + "="*80)
    print("🤖 AI-POWERED TRADING BOT v4.0 - Groq Integration".center(80))
    print("Auto-Trade berdasarkan AI Analysis".center(80))
    print("="*80 + "\n")
    
    # Load API keys
    if not GROQ_KEYS:
        Logger.error("No Groq API keys found. Create api.txt dengan API keys (one per line)")
        exit(1)
    
    # Initialize bot
    bot = AITradingBot(GROQ_KEYS, exchange, MAX_POSITIONS)
    
    # Run continuous analysis & trading
    bot.run_continuous(interval=300)  # Every 5 minutes

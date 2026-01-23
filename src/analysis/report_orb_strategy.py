import pandas as pd
from data_fetcher import fetch_data
from strategy_orb import ORBStrategy
import json
import time

def load_assets():
    with open('strategy_config.json', 'r') as f:
        config = json.load(f)
    return config['assets']

def run_backtest():
    assets = load_assets()
    
    # Filter for TradFi and BTC
    target_assets = [a for a in assets if ('=X' in a or '=F' in a or '^' in a or 'BTC' in a or '-' in a)]
    # Ensure BTC is included if strategy config uses different naming
    # Check if BTC found
    has_btc = any('BTC' in a for a in target_assets)
    if not has_btc:
        # Add BTC manually if not found (e.g. if config only has alts)
        target_assets.append('BTC/USDT')

    print(f"Testing ORB on {len(target_assets)} assets...")
    
    strat = ORBStrategy(tp_mult=2.0)
    
    results = []
    
    for symbol in target_assets:
        atype = 'crypto' if 'BTC' in symbol or 'USDT' in symbol else 'trad'
        
        # Fetch 60 days of 15m data (Max for Yahoo usually)
        # Limit 3000 candles ~ 30-40 days of 15m trading hours
        # Crypto 24/7 -> 96 candles/day -> 3000 candles = 31 days.
        # TradFi -> 26 candles/day -> 3000 candles = 115 days.
        
        try:
            print(f"Fetching {symbol}...")
            df = fetch_data(symbol, atype, '15m', limit=4000) 
            
            if df is None or df.empty:
                print(f"  No data for {symbol}")
                continue
                
            # Apply ORB
            df_res = strat.apply(df, atype)
            
            if df_res.empty or 'signal_type' not in df_res.columns:
                print(f"  No signals generated or empty result.")
                continue
                
            signals = df_res[df_res['signal_type'].notna()].copy()
            
            if signals.empty:
                print(f"  Zero ORB signals.")
                continue
                
            # Simulate Outcomes
            # Logic: Entry taken. Hold until TP/SL or End of Session.
            # Assuming ORB implies Intraday Exit.
            
            wins = 0
            losses = 0
            breakevens = 0
            total_pnl = 0.0
            
            for t, row in signals.iterrows():
                entry = row['entry_price']
                tp = row['take_profit']
                sl = row['stop_loss']
                sig = row['signal_type']
                
                # Look forward in same day
                day_start = t
                # Get end of day (approx). For crypto, end of session (24h?) ORB usually invalid after X hours.
                # Let's check pure TP/SL hit in future data
                
                future_df = df_res.loc[t:].iloc[1:]
                
                trade_pnl = 0
                outcome = "OPEN"
                
                for ft, frow in future_df.iterrows():
                    # Check stops
                    h = frow['high']
                    l = frow['low']
                    
                    if sig == 'ORB_LONG':
                        if l <= sl:
                            trade_pnl = (sl - entry)/entry
                            outcome = "LOSS"
                            break
                        if h >= tp:
                            trade_pnl = (tp - entry)/entry
                            outcome = "WIN"
                            break
                    else:
                        if h >= sl:
                            trade_pnl = (entry - sl)/entry
                            outcome = "LOSS"
                            break
                        if l <= tp:
                            trade_pnl = (entry - tp)/entry
                            outcome = "WIN"
                            break
                    
                    # Force Close at End of Day for TradFi?
                    # Check if day changed
                    if frow['session_day'] != row['session_day']:
                        # EOD Close
                        c = frow['open'] # Close at open of next day? Or close of last bar?
                        # Simplified: Force close at last bar of day.
                        # We went past day. Close at previous bar close.
                        # Approximation: Close at current Open.
                        
                        if sig == 'ORB_LONG':
                            trade_pnl = (c - entry)/entry
                        else:
                            trade_pnl = (entry - c)/entry
                        
                        outcome = "EOD"
                        break
                
                # Fee
                trade_pnl -= 0.001 
                
                total_pnl += trade_pnl
                if trade_pnl > 0: wins += 1
                else: losses += 1
            
            count = wins + losses
            wr = wins/count if count > 0 else 0
            
            print(f"  {symbol}: Trades: {count}, WR: {wr:.0%}, Net: {total_pnl:.2%}")
            
            results.append({
                "Asset": symbol,
                "Trades": count,
                "Win Rate": f"{wr:.0%}",
                "Net PnL": f"{total_pnl:.2%}"
            })
            
        except Exception as e:
            print(f"Error {symbol}: {e}")
            
    # Summary
    if results:
        res_df = pd.DataFrame(results)
        print("\n=== ORB Strategy Results (Last ~60 Days) ===")
        print(res_df.to_markdown())
        res_df.to_csv("orb_backtest_results.csv")
    else:
        print("No results.")

if __name__ == "__main__":
    run_backtest()

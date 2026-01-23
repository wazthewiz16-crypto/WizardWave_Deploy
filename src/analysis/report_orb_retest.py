import pandas as pd
from data_fetcher import fetch_data
from strategy_orb_retest import ORBStrategy_Retest
import time

def run_backtest_retest():
    # Specific Assets
    target_assets = ["BTC/USDT", "^NDX", "^GSPC", "GC=F"]
    
    print(f"Testing ORB Re-test Strategy (5m Confirmation, 3R) on {len(target_assets)} assets...")
    
    strat = ORBStrategy_Retest(session_start_est="09:30", rr_ratio=3.0)
    
    results = []
    
    for symbol in target_assets:
        atype = 'crypto' if 'BTC' in symbol else 'trad'
        
        try:
            print(f"Fetching {symbol} (5m data)...")
            # Fetch 60 days of 5m data
            df = fetch_data(symbol, atype, '5m', limit=10000) 
            
            if df is None or df.empty:
                print(f"  No data for {symbol}")
                continue
                
            # Apply ORB Retest
            df_res = strat.apply(df, atype)
            
            signals = df_res[df_res['signal_type'].notna()].copy()
            
            if signals.empty:
                print(f"  Zero Signals.")
                continue
                
            wins = 0
            losses = 0
            total_pnl = 0.0
            
            for t, row in signals.iterrows():
                entry = row['entry_price']
                tp = row['take_profit']
                sl = row['stop_loss']
                sig = row['signal_type']
                
                # Check Future Outcome
                final_pnl = 0
                outcome = "OPEN"
                
                future_df = df_res.loc[t:].iloc[1:] # 5m bars after entry
                
                for ft, frow in future_df.iterrows():
                    h = frow['high']
                    l = frow['low']
                    
                    if sig == 'ORB_LONG':
                        if l <= sl:
                            final_pnl = -1.0 # 1R Loss (Since risk is denominator)
                            # Actually, stick to % PnL for comparison
                            final_pnl = (sl - entry)/entry
                            outcome = "LOSS"
                            break
                        if h >= tp:
                            final_pnl = (tp - entry)/entry
                            outcome = "WIN"
                            break
                    else:
                        if h >= sl:
                            final_pnl = (entry - sl)/entry
                            outcome = "LOSS"
                            break
                        if l <= tp:
                            final_pnl = (entry - tp)/entry
                            outcome = "WIN"
                            break
                    
                    # EOD Close (Same Day)
                    if frow['session_day'] != row['session_day']:
                        # Close at open of next session? Or previous close?
                        # Let's say we close at the last bar of the day.
                        # Approximate using current open
                        c = frow['open']
                        if sig == 'ORB_LONG': final_pnl = (c - entry)/entry
                        else: final_pnl = (entry - c)/entry
                        outcome = "EOD"
                        break
                        
                final_pnl -= 0.001 # Fee
                
                total_pnl += final_pnl
                if final_pnl > 0: wins += 1
                else: losses += 1
                
            count = wins + losses
            wr = wins/count if count > 0 else 0
            
            print(f"  {symbol}: Trades: {count}, WR: {wr:.0%}, Net PnL: {total_pnl:.2%}")
            
            results.append({
                "Asset": symbol,
                "Trades": count,
                "Win Rate": f"{wr:.0%}",
                "Net PnL": f"{total_pnl:.2%}"
            })
            
        except Exception as e:
            print(f"Error {symbol}: {e}")
            import traceback
            traceback.print_exc()

    if results:
        res_df = pd.DataFrame(results)
        print("\n=== ORB Retest Strategy (3R) Results ===")
        print(res_df.to_markdown())
    else:
        print("No results.")

if __name__ == "__main__":
    run_backtest_retest()

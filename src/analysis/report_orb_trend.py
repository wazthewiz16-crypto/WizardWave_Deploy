import pandas as pd
from data_fetcher import fetch_data
from strategy_orb_retest import ORBStrategy_Retest
import pandas_ta as ta

def run_backtest_trend():
    assets = ["BTC/USDT", "^NDX", "^GSPC", "GC=F"]
    print(f"Testing ORB Retest (3R) with HTF Trend Filter (1H EMA200)...")
    
    strat = ORBStrategy_Retest(session_start_est="09:30", rr_ratio=3.0)
    results = []
    
    for symbol in assets:
        try:
            atype = 'crypto' if 'BTC' in symbol else 'trad'
            print(f"Processing {symbol}...")
            
            # 1. Fetch HTF Data (1H) for Trend
            # Use '1y' to ensure enough history for EMA200
            df_1h = fetch_data(symbol, atype, '1h', limit=4000)
            if df_1h is None or df_1h.empty:
                print("  No 1H data.")
                continue
                
            # Calc EMA 200
            df_1h['ema200'] = ta.ema(df_1h['close'], length=200)
            
            # Map Day -> Trend Direction (UP/DOWN/FLAT)
            # We use the trend at the START of the session (e.g. 09:00 bar or 08:00 bar)
            # Convert to local time day
            
            if df_1h.index.tz is None:
                # Assuming UTC from fetcher
                df_1h.index = df_1h.index.tz_localize('UTC').tz_convert('America/New_York')
            else:
                 df_1h.index = df_1h.index.tz_convert('America/New_York')
                 
            htf_trend_map = {}
            days = df_1h.index.date
            unique_days = sorted(list(set(days)))
            
            for d in unique_days:
                # Get last bar before 09:30? Or 09:00 bar.
                # TradFi: 09:30 Open. We can check trend at 09:00 or close of previous day.
                # Let's check trend at 09:00 bar close (if exists) or previous close.
                
                day_data = df_1h[df_1h.index.date == d]
                if day_data.empty: continue
                
                # Check 9am bar
                target_check = day_data[day_data.index.hour == 9]
                if target_check.empty:
                    # Maybe it's 24h market? Or 1h bars align differently.
                    # Or previous day close.
                    # Use last available bar before 09:30
                    mask_before = day_data.index.time <= pd.to_datetime("09:30").time()
                    subset = day_data[mask_before]
                    if subset.empty:
                        # try yesterday last bar
                        prev_day = df_1h[df_1h.index.date < d]
                        if prev_day.empty: continue
                        check_row = prev_day.iloc[-1]
                    else:
                        check_row = subset.iloc[-1]
                else:
                    check_row = target_check.iloc[0]
                
                c = check_row['close']
                ema = check_row['ema200']
                
                if pd.isna(ema): trend = "NEUTRAL"
                elif c > ema: trend = "UP"
                else: trend = "DOWN"
                
                htf_trend_map[d] = trend
            
            # 2. Fetch 5m Data
            df_5m = fetch_data(symbol, atype, '5m', limit=10000)
            if df_5m is None: continue
            
            # 3. Apply Strat
            df_res = strat.apply(df_5m, atype, htf_trend_map)
            
            # 4. Calc Stats
            signals = df_res[df_res['signal_type'].notna()]
            if signals.empty:
                print("  Zero signals after filter.")
                continue
                
            wins = 0
            losses = 0
            net = 0.0
            
            for t, row in signals.iterrows():
                # Re-run pnl logic (simplified)
                # ... reuse logic from report_orb_retest.py ...
                entry = row['entry_price']
                tp = row['take_profit']
                sl = row['stop_loss']
                sig = row['signal_type']
                
                outcome = "OPEN"
                final_pnl = 0
                future = df_res.loc[t:].iloc[1:]
                
                for ft, frow in future.iterrows():
                    h, l = frow['high'], frow['low']
                    if sig == 'ORB_LONG':
                        if l <= sl: 
                            final_pnl = (sl - entry)/entry; outcome="LOSS"; break
                        if h >= tp:
                            final_pnl = (tp - entry)/entry; outcome="WIN"; break
                    else:
                        if h >= sl:
                            final_pnl = (entry - sl)/entry; outcome="LOSS"; break
                        if l <= tp:
                            final_pnl = (entry - tp)/entry; outcome="WIN"; break
                    
                    if frow['session_day'] != row['session_day']:
                        c = frow['open']
                        if sig == 'ORB_LONG': final_pnl = (c - entry)/entry
                        else: final_pnl = (entry - c)/entry
                        outcome = "EOD"
                        break
                
                net += (final_pnl - 0.001)
                if outcome == "WIN" or (outcome == "EOD" and final_pnl > 0): 
                    # EOD wins count as wins? For 'Win Rate' strictly TP hit is usually counted
                    # but profit is profit. Let's count profitable as win.
                    if final_pnl > 0: wins += 1
                    else: losses += 1 # Breakeven is likely loss due to fee
                else:
                    if final_pnl > 0: wins += 1
                    else: losses += 1
            
            count = wins + losses
            wr = wins/count if count > 0 else 0
            
            print(f"  {symbol}: T={count}, WR={wr:.0%}, Net={net:.2%}")
            results.append({"Asset": symbol, "Trades": count, "Win Rate": f"{wr:.0%}", "Net PnL": f"{net:.2%}"})
            
        except Exception as e:
            print(f"Error {symbol}: {e}")
            import traceback
            traceback.print_exc()

    if results:
        print("\n=== ORB Retest + Trend Filter Results ===")
        print(pd.DataFrame(results).to_markdown())

if __name__ == "__main__":
    run_backtest_trend()

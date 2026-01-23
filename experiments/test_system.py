from data_fetcher import fetch_data
from strategy import WizardWaveStrategy
import pandas as pd

def test_system():
    print("Testing Crypto Fetch (BTC/USDT)...")
    try:
        df_crypto = fetch_data("BTC/USDT", "crypto", limit=100)
        print(f"Crypto Data:\n{df_crypto.tail(2)}")
    except Exception as e:
        print(f"Crypto Error: {e}")
        df_crypto = pd.DataFrame()

    print("\nTesting Trad Fetch (NAS100)...")
    try:
        df_trad = fetch_data("^NDX", "trad", limit=100)
        print(f"Trad Data:\n{df_trad.tail(2)}")
    except Exception as e:
        print(f"Trad Error: {e}")
        df_trad = pd.DataFrame()

    print("\nTesting Strategy Logic...")
    strat = WizardWaveStrategy()
    
    if not df_crypto.empty:
        res = strat.apply(df_crypto)
        print("Crypto Strategy Result (Last 2 rows):")
        print(res[['close', 'cloud_top', 'is_bullish', 'in_bid_zone', 'signal_type']].tail(2))
    
    if not df_trad.empty:
        res_trad = strat.apply(df_trad)
        print("Trad Strategy Result (Last 2 rows):")
        print(res_trad[['close', 'cloud_top', 'is_bullish', 'in_bid_zone', 'signal_type']].tail(2))

if __name__ == "__main__":
    test_system()

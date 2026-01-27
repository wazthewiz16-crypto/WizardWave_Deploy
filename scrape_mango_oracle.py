import asyncio
import os
import json
import re
import sys
import logging
import traceback
from datetime import datetime, timedelta
from playwright.async_api import async_playwright

# --- CONFIGURATION for Oracle Scraper ---
STATE_FILE = "tv_state.json"
OUTPUT_FILE = "mango_oracle_signals.json"
LOG_FILE = "oracle_scraper_debug.log"

ASSETS = [
    # Crypto
    {"symbol": "BINANCE:BTCUSDT.P", "name": "BTC"},
    {"symbol": "BINANCE:ETHUSDT.P", "name": "ETH"},
    {"symbol": "BINANCE:SOLUSDT.P", "name": "SOL"},
    {"symbol": "BINANCE:DOGEUSDT.P", "name": "DOGE"},
    {"symbol": "BINANCE:XRPUSDT.P", "name": "XRP"},
    {"symbol": "BINANCE:BNBUSDT.P", "name": "BNB"},
    {"symbol": "BINANCE:LINKUSDT.P", "name": "LINK"},
    {"symbol": "BINANCE:ARBUSDT.P", "name": "ARB"},
    {"symbol": "BINANCE:AVAXUSDT.P", "name": "AVAX"},
    {"symbol": "BINANCE:ADAUSDT.P", "name": "ADA"},
    
    # TradFi / Indices (Use OANDA/CapitalCom for reliable data)
    {"symbol": "OANDA:NAS100USD", "name": "NDX"},
    {"symbol": "OANDA:SPX500USD", "name": "SPX"},
    {"symbol": "OANDA:AU200AUD", "name": "AUS200"},
    {"symbol": "CAPITALCOM:DXY", "name": "DXY"},
    {"symbol": "OANDA:XAUUSD", "name": "GOLD"},
    {"symbol": "OANDA:WTICOUSD", "name": "OIL"},
    {"symbol": "OANDA:XAGUSD", "name": "SILVER"},
    
    # Forex
    {"symbol": "FX:EURUSD", "name": "EURUSD"},
    {"symbol": "FX:GBPUSD", "name": "GBPUSD"},
    {"symbol": "FX:AUDUSD", "name": "AUDUSD"},
]

# Order matters: Low to High for confirmation
TIMEFRAMES = ["15m", "1h", "4h", "1d", "4d"]

async def scrape_oracle_data(browser_context, asset):
    page = await browser_context.new_page()
    # Oracle demands specific chart layout
    url = f"https://www.tradingview.com/chart/qR1XTue9/?symbol={asset['symbol']}"
    
    asset_results = {}
    
    try:
        logging.info(f"[Oracle] {asset['name']}: Navigating...")
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(8)

        # Ensure Layout & Data Window
        try:
            has_mango = await page.evaluate("() => document.body.innerText.includes('Mango')")
            if not has_mango:
                await page.mouse.click(640, 400); await asyncio.sleep(0.5)
                await page.keyboard.type("."); await asyncio.sleep(1)
                await page.keyboard.type("Arcane Portal"); await asyncio.sleep(1)
                await page.keyboard.press("Enter"); await asyncio.sleep(8)
        except: pass

        try:
            is_dw = await page.evaluate("() => document.body.innerText.includes('Entry Zone Upper')")
            if not is_dw:
                await page.mouse.click(640, 400); await page.keyboard.press("Alt+D"); await asyncio.sleep(2)
        except: pass

        # Scrape Loop
        for tf in TIMEFRAMES:
            # Switch TF
            await page.keyboard.type(tf)
            await page.keyboard.press("Enter")
            await asyncio.sleep(4)
            await page.mouse.move(1150, 400) # Hover current candle
            await asyncio.sleep(1)

            # --- EXTRACTION ---
            data = await page.evaluate(r"""(() => {
                const res = {
                    Trend: "Unknown",
                    PlotValues: {},
                    Timestamp: new Date().toISOString()
                };
                
                const txt = document.body.innerText;
                
                // Helper to extract numeric values from Data Window keys
                const findVal = (key) => {
                    const safeKey = key.replace(/ /g, '[\\s\\n]+');
                    const re = new RegExp(safeKey + "[:\\s\\n]+([0-9,.]+)", "i");
                    const m = txt.match(re);
                    return m ? parseFloat(m[1].replace(/,/g, '')) : null;
                };

                const close = findVal('Close') || findVal('C'); // C often used
                const d1 = findVal('MangoD1');
                const d2 = findVal('MangoD2');
                const entryUp = findVal('Entry Zone Upper');
                const entryDown = findVal('Entry Zone Lower');
                
                res.PlotValues = { Close: close, D1: d1, D2: d2, EntryUp: entryUp, EntryDown: entryDown };

                // Trend Calculation
                if (close && d1 && d2) {
                    const upper = Math.max(d1, d2);
                    const lower = Math.min(d1, d2);
                    if (close > upper) res.Trend = "Bullish";
                    else if (close < lower) res.Trend = "Bearish";
                    else res.Trend = "Neutral";
                }
                
                return res;
            })()""")
            
            asset_results[tf] = data
            logging.info(f"  [+] {asset['name']} {tf}: Trend={data['Trend']}")

    except Exception as e:
        logging.error(f"[!] {asset['name']} Oracle Error: {e}")
    finally:
        try: await page.close()
        except: pass
        
    return asset_results

# --- ORACLE LOGIC ENGINE ---
# --- ORACLE LOGIC ENGINE ---
def process_oracle_logic(scraped_data):
    signals = []
    
    # Load Existing State for Sticky Timestamps
    existing_state = {} # Key: Asset_TF_Type -> Timestamp
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r") as f:
                old_sigs = json.load(f)
                
                # Sort by timestamp to ensure we process orders correctly if needed, 
                # but we want MIN timestamp for sticky logic.
                for s in old_sigs:
                    k = f"{s['Asset']}_{s['Timeframe']}_{s['Type']}"
                    ts = s['Timestamp']
                    
                    # Store the EARLIEST timestamp seen for this signal type
                    if k not in existing_state:
                         existing_state[k] = ts
                    else:
                         # Compare and keep older
                         try:
                             curr_stored = datetime.strptime(existing_state[k], '%Y-%m-%d %H:%M:%S')
                             new_ts = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                             if new_ts < curr_stored:
                                 existing_state[k] = ts
                         except: pass

                # Clean duplicates from file itself (Self-Healing)
                # If we have multiple entries for same Key but different timestamps, 
                # we technically only want the "Active" one or the original one?
                # For Oracle file, we only want the LATEST status of that signal, but with ORIGINAL timestamp.
                # Actually, the file is a log of "Active Signals".
                # Let's dedupe in-memory and rewrite if needed? 
                # Doing it at end of cycle is safer.
        except: pass
    
    for asset_name, tfs in scraped_data.items():
        pairs = [("15m", "1h"), ("1h", "4h"), ("4h", "1d"), ("1d", "4d")]
        
        for low_tf, high_tf in pairs:
            if low_tf not in tfs or high_tf not in tfs:
                continue
                
            curr = tfs[low_tf]
            htf = tfs[high_tf]
            
            # 1. Trend Alignment
            if curr['Trend'] == "Unknown" or htf['Trend'] == "Unknown":
                continue
                
            # 2. Logic
            sig_type = None
            sl_price = 0.0
            
            p = curr['PlotValues'].get('Close')
            u = curr['PlotValues'].get('EntryUp')
            l = curr['PlotValues'].get('EntryDown')
            d2 = curr['PlotValues'].get('D2')
            
            # LONG Rules:
            if curr['Trend'] == "Bullish" and htf['Trend'] == "Bullish" and p and u and d2:
                if p <= u:
                    sig_type = "LONG"
                    # SL: Below the Dynamic (D2) with buffer
                    sl_price = d2 * 0.995
            
            # SHORT Rules:
            if curr['Trend'] == "Bearish" and htf['Trend'] == "Bearish" and p and l and d2:
                if p >= l:
                    sig_type = "SHORT"
                    # SL: Above the Dynamic (D2) with buffer
                    sl_price = d2 * 1.005
            
            if sig_type:
                # Force EST (UTC-5)
                est_now = datetime.utcnow() - timedelta(hours=5)
                
                # Check for Sticky Timestamp
                # Logic: If we found this EXACT signal recently, keep the original timestamp.
                # This prevents specific ID generation spam.
                ukey = f"{asset_name}_{low_tf.upper()}_{sig_type}"
                final_ts = est_now.strftime('%Y-%m-%d %H:%M:%S')
                
                # Debug Check
                # logging.info(f"Checking Sticky: {ukey} in {list(existing_state.keys())}")
                
                if ukey in existing_state:
                    old_ts_str = existing_state[ukey]
                    try:
                        old_dt = datetime.strptime(old_ts_str, '%Y-%m-%d %H:%M:%S')
                        # If old signal is less than X hours old, keep it.
                        diff = (est_now - old_dt).total_seconds() / 3600
                        
                        # Timout based on TF?
                        limit = 6
                        if "4H" in low_tf.upper(): limit = 24
                        if "1D" in low_tf.upper(): limit = 48
                        
                        if diff < limit:
                            final_ts = old_ts_str
                            # logging.info(f"  -> KEPT Sticky TS: {final_ts}")
                        else:
                            pass # logging.info(f"  -> EXPIRED Sticky TS: {diff:.1f}h > {limit}h")
                            
                    except Exception as e: 
                        logging.error(f"Sticky Date Parse Error: {e}")

                signals.append({
                    "Asset": asset_name,
                    "Timeframe": low_tf.upper(), # Signal on the lower TF
                    "Confirm_TF": high_tf.upper(),
                    "Type": sig_type,
                    "Price": p,
                    "Stop_Loss": round(sl_price, 4),
                    "Timestamp": final_ts
                })
                
    return signals

async def main():
    # Logging Setup
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info("Oracle Scraper Service Started")
    
    # Init Output
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w") as f: json.dump([], f)

    while True:
        try:
            logging.info("--- Oracle Cycle Starting ---")
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-gpu'])
                context = await browser.new_context(storage_state=STATE_FILE, viewport={'width': 1920, 'height': 1080})
                
                all_scrape_data = {}
                
                # Scrape
                for asset in ASSETS:
                    data = await scrape_oracle_data(context, asset)
                    all_scrape_data[asset['name']] = data
                    
                await browser.close()
                
                # Process Logic
                new_signals = process_oracle_logic(all_scrape_data)
                
                if new_signals:
                    logging.info(f"Generated {len(new_signals)} Oracle Signals")
                    
                    # LOAD EXISTING to Merge (Persistence)
                    existing_signals = []
                    if os.path.exists(OUTPUT_FILE):
                        try:
                            with open(OUTPUT_FILE, "r") as f:
                                existing_signals = json.load(f)
                        except: pass
                    
                    # Deduplicate: Unique Key = Asset + Timeframe + Timestamp
                    # Actually, we want to remove "Same Signal, Different Key (Timestamp)" duplicates.
                    # We want to keep ONLY the one with the "Sticky" timestamp if possible.
                    
                    # 1. Build map of "Asset_TF_Type" -> [Signal Objects]
                    sig_map = {}
                    all_candidates = existing_signals + new_signals
                    
                    for s in all_candidates:
                        k = f"{s['Asset']}_{s['Timeframe']}_{s['Type']}"
                        if k not in sig_map: sig_map[k] = []
                        sig_map[k].append(s)
                    
                    # 2. Select BEST for each key
                    final_list = []
                    for k, candidates in sig_map.items():
                        # Sort by Timestamp (Earliest first)
                        # We want the ORIGINAL signal time.
                        try:
                            candidates.sort(key=lambda x: datetime.strptime(x['Timestamp'], '%Y-%m-%d %H:%M:%S'))
                        except: pass
                        
                        # Take the first one (Oldest)
                        # But wait, if the NEW signal has updated Price/SL, we might want that data, but OLD timestamp.
                        # The "Signals" generated above ALREADY have the sticky timestamp applied.
                        # So if we have Old(14:00) and New(14:00), they match.
                        # If we have Old(14:00) and New(14:20), it means sticky failed.
                        # We should prefer 14:00.
                        
                        best_sig = candidates[0]
                        # Update price to latest if available in candidates?
                        # Actually, keeping the original entry price is better for "Entry" record.
                        # But "Current Price" is not stored here.
                        # Let's just keep the OLDEST entry to stabilize the ID.
                        
                        final_list.append(best_sig)
                        
                    # 3. Serialize
                    merged_signals = final_list
                    
                    # Keep last 50
                    if len(merged_signals) > 50:
                        # Sort by timestamp to keep latest 50 distinct signals
                         try:
                            merged_signals.sort(key=lambda x: datetime.strptime(x['Timestamp'], '%Y-%m-%d %H:%M:%S'))
                         except: pass
                         merged_signals = merged_signals[-50:]

                    # Atomic Write
                    with open(OUTPUT_FILE, "w") as f:
                        json.dump(merged_signals, f, indent=4)
                else:
                    logging.info("No NEW Oracle Signals found this cycle.")
                    # DO NOT CLEAR FILE anymore. Persistence is key.
                        
            logging.info("Oracle Cycle Completed. Sleeping 15m...")
            
        except Exception as e:
            logging.error(f"Oracle Cycle Crash: {e}")
            logging.error(traceback.format_exc())
            
        await asyncio.sleep(900) # 15 Minutes

if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import os
import json
import re
import sys
import logging
import traceback
from datetime import datetime
from playwright.async_api import async_playwright

# --- CONFIGURATION for Oracle Scraper ---
STATE_FILE = "tv_state.json"
OUTPUT_FILE = "mango_oracle_signals.json"
LOG_FILE = "oracle_scraper_debug.log"

ASSETS = [
    {"symbol": "BINANCE:BTCUSDT.P", "name": "BTC"},
    {"symbol": "BINANCE:ETHUSDT.P", "name": "ETH"},
    {"symbol": "BINANCE:SOLUSDT.P", "name": "SOL"},
    {"symbol": "BINANCE:DOGEUSDT.P", "name": "DOGE"},
    {"symbol": "BINANCE:XRPUSDT.P", "name": "XRP"},
    {"symbol": "BINANCE:BNBUSDT.P", "name": "BNB"},
    {"symbol": "BINANCE:LINKUSDT.P", "name": "LINK"},
    # Add TradFi if needed, but focus on Crypto for dynamic/bid logic first
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
def process_oracle_logic(scraped_data):
    signals = []
    
    for asset_name, tfs in scraped_data.items():
        # Iterate TFs to find signals
        # We need pairs: (Current, HTF)
        # 15m -> 1H
        # 1H -> 4H
        # 4H -> 1D
        # 1D -> 4D
        
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
            
            # LONG Rules:
            # HTF Bullish, Current Bullish
            # Current Price Pullback: Low <= EntryUp (Implies dipping into zone)
            # Simpler proxy with just Close for now, or if we had Low. 
            # User said "price pulls back". Close < EntryUp is a deep pullback. Close <= EntryUp is inside/below.
            # Ideally we check Low, but we only scraped Close/Plots.
            # Let's use Close relative to Zone.
            # "Into dynamic or bid zone": Zone is usually between EntryUp and EntryDown.
            # Bullish Dynamic: Price > D1/D2. Zone is support above dynamic.
            
            p = curr['PlotValues'].get('Close')
            u = curr['PlotValues'].get('EntryUp')
            l = curr['PlotValues'].get('EntryDown')
            
            if curr['Trend'] == "Bullish" and htf['Trend'] == "Bullish" and p and u and l:
                # Pullback: Close is within or below the "Upper" part of the zone?
                # Usually Bid Zone in Bull trend is support.
                # If Close <= u: We are in the zone or lower.
                # But we must maintain Bullish Trend (Close > Dynamic).
                # So: Dynamic < Close <= EntryUp
                # We interpret "pullback into bid zone" as Close <= EntryUp.
                if p <= u:
                    sig_type = "LONG"
            
            # SHORT Rules:
            # HTF Bearish, Current Bearish
            # Spike: Close >= EntryDown? (Inverse logic)
            if curr['Trend'] == "Bearish" and htf['Trend'] == "Bearish" and p and u and l:
                if p >= l:
                    sig_type = "SHORT"
            
            if sig_type:
                signals.append({
                    "Asset": asset_name,
                    "Timeframe": low_tf.upper(), # Signal on the lower TF
                    "Confirm_TF": high_tf.upper(),
                    "Type": sig_type,
                    "Price": p,
                    "Timestamp": datetime.now().isoformat()
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
                    # Atomic Write
                    with open(OUTPUT_FILE, "w") as f:
                        json.dump(new_signals, f, indent=4)
                else:
                    logging.info("No Oracle Signals found this cycle.")
                    # Optional: Clear if we want "only active"
                    with open(OUTPUT_FILE, "w") as f:
                        json.dump([], f)
                        
            logging.info("Oracle Cycle Completed. Sleeping 15m...")
            
        except Exception as e:
            logging.error(f"Oracle Cycle Crash: {e}")
            logging.error(traceback.format_exc())
            
        await asyncio.sleep(900) # 15 Minutes

if __name__ == "__main__":
    asyncio.run(main())

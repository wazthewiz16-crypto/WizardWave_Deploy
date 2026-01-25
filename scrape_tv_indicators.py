import asyncio
import os
import json
import re
import subprocess
import sys
import logging
import traceback
from datetime import datetime

# Self-healing: Install playwright if missing (Streamlit Cloud sometimes misses requirements on fast updates)
try:
    from playwright.async_api import async_playwright
except ImportError:
    print("[!] Playwright module missing. Installing at runtime...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright==1.49.0", "greenlet==3.1.1", "pyee==12.0.0", "typing-extensions==4.15.0"])
    
    # Fix: Reload import system to see new packages
    import importlib
    import site
    importlib.invalidate_caches()
    
    # Ensure user site packages are in path (common issue in Streamlit Cloud)
    if hasattr(site, 'getusersitepackages'):
        user_site = site.getusersitepackages()
        if user_site not in sys.path:
            sys.path.append(user_site)
            print(f"[+] Added {user_site} to sys.path")
            
    from playwright.async_api import async_playwright

# Full Assets List
ASSETS = [
    {"symbol": "BINANCE:BTCUSDT", "name": "BTC"},
    {"symbol": "BINANCE:ETHUSDT", "name": "ETH"},
    {"symbol": "BINANCE:SOLUSDT", "name": "SOL"},
    {"symbol": "BINANCE:DOGEUSDT", "name": "DOGE"},
    {"symbol": "BINANCE:XRPUSDT", "name": "XRP"},
    {"symbol": "BINANCE:BNBUSDT", "name": "BNB"},
    {"symbol": "BINANCE:LINKUSDT", "name": "LINK"},
    {"symbol": "NASDAQ:NDX", "name": "NDX"},
    {"symbol": "TVC:SPX", "name": "SPX"},
    {"symbol": "ASX:XJO", "name": "AUS200"},
    {"symbol": "TVC:DXY", "name": "DXY"},
    {"symbol": "TVC:GOLD", "name": "GOLD"},
    {"symbol": "NYMEX:CL1!", "name": "OIL"},
    {"symbol": "TVC:SILVER", "name": "SILVER"},
    {"symbol": "BINANCE:ARBUSDT", "name": "ARB"},
    {"symbol": "BINANCE:AVAXUSDT", "name": "AVAX"},
    {"symbol": "BINANCE:ADAUSDT", "name": "ADA"},
    {"symbol": "CRYPTOCAP:BTC.D", "name": "BTC.D"},
    {"symbol": "CRYPTOCAP:USDT.D", "name": "USDT.D"}
]
TIMEFRAMES = ["1h", "4h", "1d"]

async def scrape_asset_data(browser_context, asset):
    page = await browser_context.new_page()
    # Using the specific chart layout ID provided by user
    url = f"https://www.tradingview.com/chart/qR1XTue9/?symbol={asset['symbol']}"
    
    results = {}
    
    try:
        logging.info(f"  [>] {asset['name']}: Navigating...")
        await page.goto(url, wait_until="load", timeout=60000)
        await asyncio.sleep(5) # Reduced from 10s
        
        # Smart Data Window Toggle
        try:
            # Check if Data Window is already open (text check)
            is_dw_open = await page.evaluate("() => document.body.innerText.includes('Mango Dynamic')")
            
            if not is_dw_open:
                logging.info(f"    [>] {asset['name']}: Data Window closed. Toggling Alt+D...")
                await page.mouse.click(640, 400) # Center
                await asyncio.sleep(0.5)
                await page.keyboard.press("Alt+D")
                await asyncio.sleep(2)
            else:
                 logging.info(f"    [>] {asset['name']}: Data Window already valid.")
                 
        except Exception as e:
            logging.error(f"    [!] {asset['name']}: DW Toggle failed: {e}")

        await asyncio.sleep(1)

        for tf in TIMEFRAMES:
            # print(f"    [-] Switching to {tf.upper()}...")
            await page.keyboard.type(tf)
            await page.keyboard.press("Enter")
            await asyncio.sleep(4) # Reduced from 10s

            # Advanced Extraction Logic: Parse the Data Window
            data = await page.evaluate(r"""(() => {
                const results = {
                    Trend: "Unknown",
                    Tempo: "Unknown",
                    "Bid Zone": "Unknown",
                    PlotValues: {}
                };

                // Find the Data Window content wrapper
                const wrappers = Array.from(document.querySelectorAll('div')).filter(el => 
                    el.innerText && el.innerText.includes('Mango Dynamic')
                );

                if (wrappers.length > 0 || document.body.innerText.includes('Mango Dynamic')) {
                    // Get all text lines in the Data Window (or whole body if specific wrapper not found)
                    const textLines = document.body.innerText.split('\n');
                    
                    // 1. Try to find explicit status first
                    const trendIdx = textLines.findIndex(l => l.includes('Trend:'));
                    if (trendIdx !== -1) {
                         results.Trend = textLines[trendIdx].split(':')[1].trim();
                    }
                    
                    const tempoIdx = textLines.findIndex(l => l.includes('Tempo:'));
                    if (tempoIdx !== -1) {
                         results.Tempo = textLines[tempoIdx].split(':')[1].trim();
                    }
                    
                    // Improved Bid Zone Extraction (Handles same-line and next-line values)
                    const bidIdx = textLines.findIndex(l => l.includes('Bid Zone'));
                    if (bidIdx !== -1) {
                        const currentLine = textLines[bidIdx];
                        const nextLine = textLines[bidIdx + 1] || "";
                        
                        // 1. Check current line (e.g. "Bid Zone: Yes")
                        if (currentLine.includes("Yes")) results["Bid Zone"] = "Yes";
                        else if (currentLine.includes("No")) results["Bid Zone"] = "No";
                        
                        // 2. Check next line (Standard Data Window format: Key \n Value)
                        else if (results["Bid Zone"] === "Unknown") {
                             if (nextLine.trim() === "Yes" || nextLine.includes("Yes")) results["Bid Zone"] = "Yes";
                             else if (nextLine.trim() === "No" || nextLine.includes("No")) results["Bid Zone"] = "No";
                        }
                    }
                    
                    // Fallback: Check Legend specifically
                    if (results["Bid Zone"] === "Unknown") {
                         const legendText = document.body.innerText; // Global search fallback
                         if (legendText.includes('Bid Zone: Yes')) results["Bid Zone"] = "Yes";
                         else if (legendText.includes('Bid Zone: No')) results["Bid Zone"] = "No";
                    }

                    // 2. Extract Numerical Plot Values as fallback/confirmation
                    const findValue = (key) => {
                        // Find index of line containing key
                        const idx = textLines.findIndex(l => l.includes(key));
                        if (idx !== -1) {
                            // Try same line first "Key: Value"
                            if (textLines[idx].includes(':')) {
                                const val = parseFloat(textLines[idx].split(':')[1].replace(/,/g, ''));
                                if (!isNaN(val)) return val;
                            }
                            // Try next line "Key \n Value"
                            if (textLines[idx+1]) {
                                const val = parseFloat(textLines[idx+1].replace(/,/g, ''));
                                if (!isNaN(val)) return val;
                            }
                        }
                        return null;
                    };

                    results.PlotValues.Close = findValue('Close');
                    results.PlotValues.MangoD1 = findValue('MangoD1');
                    results.PlotValues.MangoD2 = findValue('MangoD2');
                    results.PlotValues.EntryUpper = findValue('Entry Zone Upper');
                    results.PlotValues.EntryLower = findValue('Entry Zone Lower');
                    
                    // Manual Calculation: Trend (Fallback)
                    if (results.Trend === "Unknown" && results.PlotValues.Close && results.PlotValues.MangoD1) {
                        const price = results.PlotValues.Close;
                        const d1 = results.PlotValues.MangoD1;
                        const d2 = results.PlotValues.MangoD2 || d1;
                        
                        const upper = Math.max(d1, d2);
                        const lower = Math.min(d1, d2);
                        
                        if (price > upper) results.Trend = "Bullish";
                        else if (price < lower) results.Trend = "Bearish";
                        else results.Trend = "Neutral"; 
                    }

                    // Manual Calculation: Bid Zone (Fallback)
                    if (results["Bid Zone"] === "Unknown" && results.PlotValues.EntryUpper && results.PlotValues.EntryLower && results.PlotValues.Close) {
                         const p = results.PlotValues.Close;
                         const upper = results.PlotValues.EntryUpper;
                         const lower = results.PlotValues.EntryLower;
                         
                         // Check if price is within the zone (Green Zone)
                         // Usually Bid Zone means Price is inside the buy zone?
                         // Assuming Upper > Lower
                         if (p <= upper && p >= lower) {
                              results["Bid Zone"] = "Yes";
                         } else {
                              results["Bid Zone"] = "No";
                         }
                    }
                    
                    // Fallback search keywords
                    if (results.Trend === "Unknown") {
                        if (document.body.innerText.includes("Bullish")) results.Trend = "Bullish";
                        else if (document.body.innerText.includes("Bearish")) results.Trend = "Bearish";
                    }
                }
                return results;
            })()""")
            
            data["Timestamp"] = datetime.now().isoformat()
            results[tf] = data
            # Log the extracted data for debugging
            if data['Trend'] == 'Unknown' or data['Bid Zone'] == 'Unknown':
                 logging.warning(f"    [?] {asset['name']} {tf}: Trend={data['Trend']}, BidZone={data['Bid Zone']}")
            else:
                 logging.info(f"    [=] {asset['name']} {tf}: Trend={data['Trend']}, BidZone={data['Bid Zone']}")

    except Exception as e:
        logging.error(f"  [!] {asset['name']} Error: {e}")
        try:
            await page.screenshot(path=f"debug_error_{asset['name']}.png")
        except:
            pass
    finally:
        try:
            await page.close()
        except: pass
    
    return results

async def main():
    STATE_FILE = "tv_state.json"
    OUTPUT_FILE = "mango_dynamic_data.json"
    LOG_FILE = "scraper_debug.log"

    # Setup Logging with Traceback
    import logging
    import traceback
    
    # Redirect stderr to log file for full crash capture
    sys.stderr = open(LOG_FILE, 'a')

    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info("Scraper Script Started")

    try:
        if not os.path.exists(STATE_FILE):
            logging.error(f"{STATE_FILE} not found. Run setup_tv_session.py.")
            return

        # Create Initial Output if missing
        if not os.path.exists(OUTPUT_FILE):
             with open(OUTPUT_FILE, "w") as f:
                 json.dump({}, f)

        print("--- Starting Automated Mango Scraper (30m Interval) ---")
        logging.info("Starting Automated Mango Scraper (30m Interval)")

        # Ensure browsers are installed (Critical for Streamlit Cloud)
        try:
            print("[*] Checking/Installing Playwright Browsers...")
            # Only install chromium to save time/space
            subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
        except Exception as e:
            logging.warning(f"Failed to run playwright install: {e}")

        while True:
            start_time = datetime.now()
            logging.info(f"Cycle Started: {start_time}")
            
            async with async_playwright() as p:
                try:
                    # Launching headless with robust cloud settings
                    browser = await p.chromium.launch(
                        headless=True,
                        args=[
                            '--disable-dev-shm-usage', # Crucial for Docker/Cloud
                            '--no-sandbox',
                            '--disable-setuid-sandbox',
                            '--disable-gpu',
                            '--disable-software-rasterizer',
                            '--window-size=1280,800'
                        ]
                    )
                    context = await browser.new_context(
                        storage_state=STATE_FILE,
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        viewport={'width': 1280, 'height': 800},
                        ignore_https_errors=True
                    )
                    
                    # Process Assets Concurrently (Max 3 workers)
                    all_results = {}
                    if os.path.exists(OUTPUT_FILE):
                         try:
                             with open(OUTPUT_FILE, "r") as f:
                                 all_results = json.load(f)
                         except: pass

                    semaphore = asyncio.Semaphore(3)
                    
                    async def worker(asset):
                        async with semaphore:
                            logging.info(f"Worker start: {asset['name']}")
                            try:
                                asset_data = await scrape_asset_data(context, asset)
                                if asset_data:
                                    # Update results dict (Thread-safe in simple asyncio loop)
                                    all_results[asset['name']] = asset_data
                                    # Incremental Save (Blocking IO but safe enough here)
                                    with open(OUTPUT_FILE, "w") as f:
                                        json.dump(all_results, f, indent=4)
                                    logging.info(f"Worker done: {asset['name']}")
                            except Exception as e:
                                logging.error(f"Worker failed {asset['name']}: {e}")

                    tasks = [worker(asset) for asset in ASSETS]
                    await asyncio.gather(*tasks)
                        
                    await browser.close()
                    logging.info("Cycle Completed Successfully")

                except Exception as cycle_e:
                    logging.error(f"Cycle Failed: {cycle_e}")
                    logging.error(traceback.format_exc())

            # Sleep Logic
            elapsed = (datetime.now() - start_time).total_seconds()
            sleep_sec = max(300, 1800 - elapsed)
            logging.info(f"Sleeping for {sleep_sec/60:.1f} minutes...")
            await asyncio.sleep(sleep_sec)

    except Exception as e:
        logging.critical(f"FATAL ERROR: {e}")
        logging.critical(traceback.format_exc())
        # Re-raise to crash process so it can be restarted if needed, 
        # but we wanted to log it first.
        sys.exit(1) 




if __name__ == "__main__":
    asyncio.run(main())

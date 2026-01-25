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
        # Use 'domcontentloaded' instead of 'load' to prevent timeouts on heavy TV scripts
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(5) # Reduced from 10s

        # Ensure 'Arcane Portal' Layout (User Request)
        try:
            # Check if we have the right indicators. If not, switch layout.
            has_mango = await page.evaluate("() => document.body.innerText.includes('Mango')")
            
            if not has_mango:
                logging.info(f"    [>] {asset['name']}: 'Mango' indicators missing. Attempting layout switch to 'Arcane Portal'...")
                # Click chart to focus first
                await page.mouse.click(640, 400)
                await asyncio.sleep(0.5)
                
                # Open Layout Load Dialog
                await page.keyboard.type(".") 
                await asyncio.sleep(2)
                
                # Type Layout Name
                await page.keyboard.type("Arcane Portal")
                await asyncio.sleep(2)
                
                # Select and Load
                await page.keyboard.press("Enter")
                logging.info(f"    [>] {asset['name']}: Selected 'Arcane Portal'. Waiting for reload...")
                await asyncio.sleep(8) 
        except Exception as e:
            logging.warning(f"    [!] {asset['name']}: Layout switch failed: {e}")
        
        # Smart Data Window Toggle
        try:
            # Check if Data Window is truly open by looking for specific indicator keys
            # "Data Window" text is present even in tooltips, so we look for "Entry Zone Upper"
            is_dw_open = await page.evaluate("() => document.body.innerText.includes('Entry Zone Upper')")
            
            if not is_dw_open:
                logging.info(f"    [>] {asset['name']}: Indicator data missing. Toggling Alt+D...")
                await page.mouse.click(640, 400) # Center
                await asyncio.sleep(0.5)
                await page.keyboard.press("Alt+D")
                await asyncio.sleep(2)
            else:
                 logging.info(f"    [>] {asset['name']}: Data Window active and indicator found.")
                 
        except Exception as e:
            logging.error(f"    [!] {asset['name']}: DW Toggle failed: {e}")

        await asyncio.sleep(1)

        for tf in TIMEFRAMES:
            # print(f"    [-] Switching to {tf.upper()}...")
            await page.keyboard.type(tf)
            await page.keyboard.press("Enter")
            await asyncio.sleep(3) # Wait for chart to load

            # CRITICAL: Hover over the latest candle (Right side of chart)
            # Data Window shows values for the cursor position.
            # If we don't hover right, we might read old data or nothing.
            await page.mouse.move(1150, 400) 
            await asyncio.sleep(1)

            # Retry mechanism for extraction
            max_retries = 3
            data = None
            
            for attempt in range(max_retries):
                # Wiggle mouse slightly on retries to force update
                if attempt > 0:
                    await page.mouse.move(1150 + (attempt * 10), 400 + (attempt * 10))
                    await asyncio.sleep(1)

                data = await page.evaluate(r"""(() => {
                    const results = {
                        Trend: "Unknown",
                        Tempo: "Unknown",
                        "Bid Zone": "Unknown",
                        PlotValues: {},
                        DebugRaw: []
                    };
                    
                    // Allow time for DOM to update? No, evaluate is instant.
                    // Get all text lines.
                    const rawText = document.body.innerText;
                    const textLines = rawText.split('\n').map(l => l.trim()).filter(l => l.length > 0);
                    
                    // 1. Text Parsing
                    const trendIdx = textLines.findIndex(l => l.includes('Trend:'));
                    if (trendIdx !== -1) results.Trend = textLines[trendIdx].split(':')[1].trim();
                    
                    const tempoIdx = textLines.findIndex(l => l.includes('Tempo:'));
                    if (tempoIdx !== -1) results.Tempo = textLines[tempoIdx].split(':')[1].trim();

                    // Bid Zone Text
                    const bidIdx = textLines.findIndex(l => l.includes('Bid Zone'));
                    if (bidIdx !== -1) {
                         const current = textLines[bidIdx];
                         if (current.includes('Yes')) results["Bid Zone"] = "Yes";
                         else if (current.includes('No')) results["Bid Zone"] = "No";
                         else {
                             const next = textLines[bidIdx + 1] || "";
                             if (next.includes('Yes')) results["Bid Zone"] = "Yes";
                             else if (next.includes('No')) results["Bid Zone"] = "No";
                         }
                    }

                    // 2. Numerical Plots
                    const findValue = (key) => {
                        const idx = textLines.findIndex(l => l.includes(key));
                        if (idx !== -1) {
                            if (textLines[idx].includes(':')) {
                                const val = parseFloat(textLines[idx].split(':')[1].replace(/,/g, '').trim());
                                if (!isNaN(val)) return val;
                            }
                            if (textLines[idx+1]) {
                                const val = parseFloat(textLines[idx+1].replace(/,/g, '').trim());
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
                    
                    // 3. Fallback Calculations
                    // Trend
                    if (results.Trend === "Unknown" && results.PlotValues.Close && results.PlotValues.MangoD1) {
                        const p = results.PlotValues.Close;
                        const d1 = results.PlotValues.MangoD1;
                        const d2 = results.PlotValues.MangoD2 || d1;
                        const upper = Math.max(d1, d2);
                        const lower = Math.min(d1, d2);
                        if (p > upper) results.Trend = "Bullish";
                        else if (p < lower) results.Trend = "Bearish";
                        else results.Trend = "Neutral";
                    }
                    
                    // Bid Zone
                    if (results["Bid Zone"] === "Unknown" && results.PlotValues.EntryUpper && results.PlotValues.EntryLower && results.PlotValues.Close) {
                         const p = results.PlotValues.Close;
                         const u = results.PlotValues.EntryUpper;
                         const l = results.PlotValues.EntryLower;
                         if (p <= u && p >= l) results["Bid Zone"] = "Yes (Calc)";
                         else results["Bid Zone"] = "No (Calc)";
                    }
                    
                    // Debug Dump
                    if (results.Trend === "Unknown" || results["Bid Zone"] === "Unknown") {
                        results.DebugRaw = textLines.slice(0, 50);
                    }

                    return results;
                })()""")

                # Check if we got valid data
                if data['Trend'] != "Unknown":
                    break # Success
            
            data["Timestamp"] = datetime.now().isoformat()
            results[tf] = data
            
            # Log results and RAW DEBUG if missing
            if data['Trend'] == 'Unknown' or data['Bid Zone'] == 'Unknown':
                 logging.warning(f"    [?] {asset['name']} {tf}: Trend={data['Trend']}, BidZone={data['Bid Zone']}")
                 
                 # Screenshot on failure (User Request: "what its look at")
                 try:
                     safe_tf = tf.replace("/","_")
                     await page.screenshot(path=f"debug_view_{asset['name']}_{safe_tf}.png")
                     logging.info(f"       [+] Saved debug screenshot: debug_view_{asset['name']}_{safe_tf}.png")
                 except: pass

                 if data.get('DebugRaw'):
                     logging.warning(f"       [RAW DEBUG] {data['DebugRaw'][:5]} ... (check log for full dump)")
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
                            '--window-size=1920,1080'
                        ]
                    )
                    context = await browser.new_context(
                        storage_state=STATE_FILE,
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        viewport={'width': 1920, 'height': 1080},
                        ignore_https_errors=True
                    )
                    
                    # Process Assets Concurrently (Max 3 workers)
                    all_results = {}
                    if os.path.exists(OUTPUT_FILE):
                         try:
                             with open(OUTPUT_FILE, "r") as f:
                                 all_results = json.load(f)
                         except: pass

                    # Switch to sequential (Semaphore 1) to resolve "stuck" logs and timeouts
                    semaphore = asyncio.Semaphore(1)
                    
                    async def worker(asset):
                        async with semaphore:
                            logging.info(f"Worker start: {asset['name']}")
                            try:
                                asset_data = await scrape_asset_data(context, asset)
                                if asset_data:
                                    all_results[asset['name']] = asset_data
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

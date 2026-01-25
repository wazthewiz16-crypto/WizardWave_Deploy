import asyncio
import os
import json
import re
import subprocess
import sys
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
        print(f"  [>] Navigating to {asset['symbol']}...")
        await page.goto(url, wait_until="load", timeout=90000)
        await asyncio.sleep(10) # Initial heavy load
        
        # Explicitly Open Data Window (User Request: Alt+D)
        try:
            print("    [>] Toggling Data Window via Hotkey (Alt+D)...")
            # Focus on chart area first (center of screen approx)
            await page.mouse.click(500, 300) 
            await asyncio.sleep(1)
            
            # Press hotkey to toggle ON (if off)
            # We do it twice just in case? No, toggle might close it if open.
            # Best verify if it's open? Hard to check computed style loosely.
            # We assume it starts closed or state persists.
            # User specifically asked for this.
            await page.keyboard.press("Alt+D")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"    [!] Hotkey failed: {e}")

        await asyncio.sleep(2)
        await asyncio.sleep(2)

        for tf in TIMEFRAMES:
            print(f"    [-] Switching to {tf.upper()}...")
            await page.keyboard.type(tf)
            await page.keyboard.press("Enter")
            await asyncio.sleep(10) # Give more time for the indicator and Data Window to sync

            # Advanced Extraction Logic: Parse the Data Window
            data = await page.evaluate("""() => {
                const results = {
                    Trend: "Unknown",
                    Tempo: "Unknown",
                    "Bid Zone": "Unknown",
                    PlotValues: {}
                };

                // Find the Data Window content wrapper
                // TV obfuscates classes, so we look for structure or text content
                const wrappers = Array.from(document.querySelectorAll('div')).filter(el => 
                    el.innerText && el.innerText.includes('Mango Dynamic')
                );

                if (wrappers.length > 0) {
                    // Get all text lines in the Data Window
                    const textLines = document.body.innerText.split('\\n');
                    
                    // 1. Try to find explicit status first
                    const trendIdx = textLines.findIndex(l => l.includes('Trend:'));
                    if (trendIdx !== -1) {
                         results.Trend = textLines[trendIdx].split(':')[1].trim();
                    }
                    
                    const tempoIdx = textLines.findIndex(l => l.includes('Tempo:'));
                    if (tempoIdx !== -1) {
                         results.Tempo = textLines[tempoIdx].split(':')[1].trim();
                    }
                    
                    const bidLine = textLines.find(l => l.includes('Bid Zone') || l.includes('Bid Zone:'));
                    if (bidLine) {
                         // Attempt split by colon
                         const parts = bidLine.split(':');
                         if (parts.length > 1) {
                             results["Bid Zone"] = parts[1].trim();
                         } else {
                             // Sometimes it's "Bid Zone Yes" without colon in Legend
                             if (bidLine.includes('Yes')) results["Bid Zone"] = "Yes";
                             else if (bidLine.includes('No')) results["Bid Zone"] = "No";
                         }
                    }
                    
                    // Fallback: Check Legend specifically (often has class 'title-wrapper' or similar, but text search is safer)
                    if (results["Bid Zone"] === "Unknown") {
                         const legendText = document.body.innerText; # Global search fallback
                         if (legendText.includes('Bid Zone: Yes')) results["Bid Zone"] = "Yes";
                         else if (legendText.includes('Bid Zone: No')) results["Bid Zone"] = "No";
                    }

                    // 2. Extract Numerical Plot Values as fallback/confirmation
                    // We look for the main price (Close) and the Mango levels
                    const findValue = (key) => {
                        const idx = textLines.findIndex(l => l.trim() === key);
                        if (idx !== -1 && textLines[idx+1]) {
                            return parseFloat(textLines[idx+1].replace(/,/g, ''));
                        }
                        return null;
                    };

                    results.PlotValues.Close = findValue('Close');
                    results.PlotValues.MangoD1 = findValue('MangoD1');
                    results.PlotValues.MangoD2 = findValue('MangoD2');
                    
                    // If Trend is still unknown but we have plots, calculate it
                    if (results.Trend === "Unknown" && results.PlotValues.Close && results.PlotValues.MangoD1) {
                        const price = results.PlotValues.Close;
                        const d1 = results.PlotValues.MangoD1;
                        const d2 = results.PlotValues.MangoD2 || d1;
                        
                        const upper = Math.max(d1, d2);
                        const lower = Math.min(d1, d2);
                        
                        // Strict Cloud Logic
                        if (price > upper) results.Trend = "Bullish";
                        else if (price < lower) results.Trend = "Bearish";
                        else results.Trend = "Neutral"; // Inside the cloud/channel
                    }
                    
                    // Fallback search for Bullish/Bearish keywords if logic fails
                    if (results.Trend === "Unknown") {
                        if (document.body.innerText.includes("Bullish")) results.Trend = "Bullish";
                        else if (document.body.innerText.includes("Bearish")) results.Trend = "Bearish";
                    }
                }
                return results;
            }""")
            
            data["Timestamp"] = datetime.now().isoformat()
            results[tf] = data
            print(f"    [=] Result: {data['Trend']} (Close: {data.get('PlotValues', {}).get('Close')})")

    except Exception as e:
        print(f"  [!] Error scraping {asset['name']}: {e}")
        try:
            await page.screenshot(path=f"debug_error_{asset['name']}.png")
        except:
            pass
    finally:
        # Check if we got valid data, if not, screenshot state
        try:
            has_data = False
            for tf in TIMEFRAMES:
                if results.get(tf, {}).get("Trend") != "Unknown":
                    has_data = True
                    break
            
            if not has_data:
                print(f"  [?] No data found for {asset['name']}, saving debug screenshot...")
                await page.screenshot(path=f"debug_empty_{asset['name']}.png")
        except:
            pass
            
        await page.close()
    
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
                    
                    # Process Assets
                    all_results = {}
                    if os.path.exists(OUTPUT_FILE):
                         try:
                             with open(OUTPUT_FILE, "r") as f:
                                 all_results = json.load(f)
                         except: pass

                    for asset in ASSETS:
                        logging.info(f"Fetching {asset['name']} ({asset['symbol']})...")
                        asset_data = await scrape_asset_data(context, asset)
                        if asset_data:
                            all_results[asset['name']] = asset_data
                            # Incremental Save
                            with open(OUTPUT_FILE, "w") as f:
                                json.dump(all_results, f, indent=4)
                        
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

import asyncio
import os
import json
import re
from playwright.async_api import async_playwright
from datetime import datetime

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
        
        # Ensure Data Window is open (it is often closed by default)
        # Attempt to click Data Window tab button if not open
        await page.evaluate("""() => {
            const dataWindowBtn = document.querySelector('[data-name="data-window"]');
            if (dataWindowBtn && dataWindowBtn.getAttribute('aria-selected') !== 'true') {
                dataWindowBtn.click();
            }
        }""")
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
                        
                        if (price > Math.max(d1, d2)) results.Trend = "Bullish";
                        else if (price < Math.min(d1, d2)) results.Trend = "Bearish";
                        else results.Trend = "Neutral";
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
    finally:
        await page.close()
    
    return results

async def main():
    STATE_FILE = "tv_state.json"
    OUTPUT_FILE = "mango_dynamic_data.json"

    if not os.path.exists(STATE_FILE):
        print(f"[ERROR] {STATE_FILE} not found. Run setup_tv_session.py.")
        return

    print("--- Starting Automated Mango Scraper (30m Interval) ---")

    # Ensure browsers are installed (Critical for Streamlit Cloud)
    import subprocess
    import sys
    try:
        print("[*] Checking/Installing Playwright Browsers...")
        # Only install chromium to save time/space
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
    except Exception as e:
        print(f"[!] Warning: Failed to run playwright install: {e}")
    
    while True:
        start_time = datetime.now()
        print(f"\n[!] Cycle Started: {start_time}")
        
        async with async_playwright() as p:
            try:
                # Launching headless but with a real user agent
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    storage_state=STATE_FILE,
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    viewport={'width': 1920, 'height': 1080}
                )
                
                # Load existing results to update incrementally
                all_results = {}
                if os.path.exists(OUTPUT_FILE):
                    try:
                        with open(OUTPUT_FILE, "r") as f:
                            all_results = json.load(f)
                    except:
                        pass
                
                # We process assets one by one
                for asset in ASSETS:
                    print(f"> Fetching {asset['name']} ({asset['symbol']})...")
                    asset_data = await scrape_asset_data(context, asset)
                    if asset_data:
                        all_results[asset['name']] = asset_data
                        # Save each asset to ensure we don't lose data on crash
                        with open(OUTPUT_FILE, "w") as f:
                            json.dump(all_results, f, indent=4)
                
                print(f"\n[SUCCESS] Cycle Completed at {datetime.now()}")
                await browser.close()
            except Exception as e:
                print(f"[CRITICAL ERROR] Automation Cycle Failed: {e}")
        
        # Calculate sleep to hit exactly 30 mins from start
        elapsed = (datetime.now() - start_time).total_seconds()
        sleep_sec = max(300, 1800 - elapsed) # Wait at least 5 mins before next run regardless
        print(f"[*] Sleeping for {sleep_sec/60:.1f} minutes...")
        await asyncio.sleep(sleep_sec)

if __name__ == "__main__":
    asyncio.run(main())

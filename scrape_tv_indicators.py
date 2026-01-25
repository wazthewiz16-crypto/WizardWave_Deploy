import asyncio
import os
import json
import re
import subprocess
import sys
from datetime import datetime

# Library handles via requirements.txt
try:
    from playwright.async_api import async_playwright
except ImportError:
    print("[!] Playwright module missing. Please ensure it is in requirements.txt")
    sys.exit(1)

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
    {"symbol": "BINANCE:ADAUSDT", "name": "ADA"}
]
TIMEFRAMES = ["4h", "12h", "1d", "4d", "1w"]

async def scrape_asset_data(browser_context, asset):
    page = await browser_context.new_page()
    # Using the specific chart layout ID provided by user
    url = f"https://www.tradingview.com/chart/qR1XTue9/?symbol={asset['symbol']}"
    
    results = {}
    
    try:
        print(f"  [>] Navigating to {asset['symbol']}...")
        await page.goto(url, wait_until="load", timeout=90000)
        await asyncio.sleep(12) # Initial heavy load
        
        # Focus chart
        await page.mouse.click(600, 400) 
        await asyncio.sleep(1)

        for tf in TIMEFRAMES:
            print(f"    [-] Switching to {tf.upper()}...")
            await page.keyboard.press("Escape") # Clear any menus
            await asyncio.sleep(0.5)
            await page.keyboard.type(tf)
            await page.keyboard.press("Enter")
            await asyncio.sleep(8) # Sync

            # Ensure Data Window is open via Hotkey
            await page.keyboard.press("Alt+D")
            await asyncio.sleep(2)

            data = await page.evaluate("""() => {
                const results = {
                    Trend: "Unknown",
                    Tempo: "Unknown",
                    "Bid Zone": "Unknown",
                    PlotValues: {},
                    StrategyState: "Neutral"
                };

                const getVal = (label) => {
                    const elements = Array.from(document.querySelectorAll('div, span, transition-group'));
                    const labelEl = elements.find(el => {
                        const t = el.innerText || "";
                        const cleanT = t.trim().toUpperCase();
                        return cleanT === label.toUpperCase() || cleanT === (label.toUpperCase() + ":");
                    });

                    if (labelEl) {
                        const parentText = labelEl.parentElement.innerText;
                        const lines = parentText.split('\\n').map(l => l.trim());
                        const idx = lines.findIndex(l => l.toUpperCase().includes(label.toUpperCase()));
                        if (idx !== -1 && lines[idx+1]) return lines[idx+1];
                        if (labelEl.innerText.includes(':')) return labelEl.innerText.split(':')[1].trim();
                    }
                    
                    const bodyLines = document.body.innerText.split('\\n').map(l => l.trim());
                    const lineIdx = bodyLines.findIndex(l => l.toUpperCase().includes(label.toUpperCase()));
                    if (lineIdx !== -1) {
                        const line = bodyLines[lineIdx];
                        if (line.includes(':')) return line.split(':')[1].trim();
                        if (bodyLines[lineIdx + 1]) return bodyLines[lineIdx + 1];
                    }
                    return null;
                };

                const findNum = (label) => {
                    const v = getVal(label);
                    if (v) {
                        const n = parseFloat(v.replace(/[^0-9.-]/g, ''));
                        return isNaN(n) ? null : n;
                    }
                    return null;
                };

                // Core Extraction
                results.Trend = getVal('Trend') || "Unknown";
                results.Tempo = getVal('Tempo') || "Unknown";
                results["Bid Zone"] = getVal('Bid Zone') || (document.body.innerText.match(/Bid Zone[:\s]+(Yes|No)/i) || [])[1] || "Unknown";

                // Prices & indicator plots
                results.PlotValues.Close = findNum('Close');
                results.PlotValues.MangoD1 = findNum('MangoD1') || findNum('Mango D1') || findNum('D1');
                results.PlotValues.MangoD2 = findNum('MangoD2') || findNum('Mango D2') || findNum('D2');
                results.PlotValues.ZoneUpper = findNum('Zone Upper') || findNum('Upper Zone');
                results.PlotValues.ZoneLower = findNum('Zone Lower') || findNum('Lower Zone');

                // TREND LOGIC (If indicator text fails)
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

                // WIZARD WAVE LOGIC (Condition 1 & 2 from Arcane Portal)
                const price = results.PlotValues.Close;
                if (price && results.PlotValues.MangoD1) {
                    const d1 = results.PlotValues.MangoD1;
                    const d2 = results.PlotValues.MangoD2 || d1;
                    const upperWave = Math.max(d1, d2);
                    const lowerWave = Math.min(d1, d2);
                    const inWave = (price <= upperWave && price >= lowerWave);
                    const isAbove = (price > upperWave);
                    const inBidZone = (results["Bid Zone"] === "Yes" || results["Bid Zone"] === "Yes:");

                    if (results.Trend === "Bullish") {
                        if (isAbove && inBidZone) results.StrategyState = "LONG_CONTINUATION";
                        else if (inWave && inBidZone) results.StrategyState = "LONG_PULLBACK";
                    } else if (results.Trend === "Bearish") {
                         const isBelow = (price < lowerWave);
                         if (isBelow && inBidZone) results.StrategyState = "SHORT_CONTINUATION";
                         else if (inWave && inBidZone) results.StrategyState = "SHORT_RECOVERY";
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
            await page.screenshot(path=os.path.join("data", "reports", f"debug_error_{asset['name']}.png"))
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
                await page.screenshot(path=os.path.join("data", "reports", f"debug_empty_{asset['name']}.png"))
        except:
            pass
            
        await page.close()
    
    return results

async def main():
    STATE_FILE = os.path.join("data", "tv_state.json")
    OUTPUT_FILE = os.path.join("data", "mango_dynamic_data.json")

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
                # Launching headless with robust cloud settings
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        '--disable-dev-shm-usage', # Crucial for Docker/Cloud
                        '--no-sandbox', 
                        '--disable-setuid-sandbox',
                        '--disable-gpu',
                        '--disable-software-rasterizer',
                        # '--single-process' # Removed as it causes instability
                    ]
                )
                context = await browser.new_context(
                    storage_state=STATE_FILE,
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    viewport={'width': 1440, 'height': 900}, # Expanded to ensure toolbars don't collapse
                    ignore_https_errors=True
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

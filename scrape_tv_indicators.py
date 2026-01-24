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
        
        # Explicitly Open Data Window (Critical Fix)
        # Explicitly Open Data Window using Hotkey (Shift + D) - Most Reliable Method
        try:
            print("    [>] Toggling Data Window via Hotkey (Alt+D)...")
            # Focus on chart area first
            await page.mouse.click(500, 300) 
            await asyncio.sleep(1)
            
            # Press hotkey
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

                // Find the Data Window content or legend
                const bodyText = document.body.innerText;
                const hasMango = /Mango/i.test(bodyText) || /Dynamic/i.test(bodyText);

                if (hasMango) {
                    // Get all text lines
                    const textLines = bodyText.split('\\n');
                    
                    // Helper to find value by dynamic partial key match
                    const getValByLabel = (label) => {
                         const line = textLines.find(l => l.toUpperCase().includes(label.toUpperCase()));
                         if (line && line.includes(':')) {
                             return line.split(':')[1].trim();
                         }
                         return null;
                    };

                    results.Trend = getValByLabel('Trend') || "Unknown";
                    results.Tempo = getValByLabel('Tempo') || "Unknown";
                    
                    // Bid Zone specific (more complex)
                    const bidVal = getValByLabel('Bid Zone');
                    if (bidVal) {
                         results["Bid Zone"] = bidVal;
                    } else {
                         // Search line by line for just the text
                         const bidIdx = textLines.findIndex(l => l.includes('Bid Zone'));
                         if (bidIdx !== -1) {
                             const nextLine = textLines[bidIdx + 1];
                             if (nextLine === 'Yes' || nextLine === 'No') results["Bid Zone"] = nextLine;
                             else if (textLines[bidIdx].includes('Yes')) results["Bid Zone"] = "Yes";
                             else if (textLines[bidIdx].includes('No')) results["Bid Zone"] = "No";
                         }
                    }
                    
                    // Fallback for Bid Zone Search in entire body
                    if (results["Bid Zone"] === "Unknown") {
                         if (bodyText.includes('Bid Zone: Yes') || bodyText.includes('Bid Zone Yes')) results["Bid Zone"] = "Yes";
                         else if (bodyText.includes('Bid Zone: No') || bodyText.includes('Bid Zone No')) results["Bid Zone"] = "No";
                    }

                    // 2. Extract Numerical Plot Values as fallback
                    const findValue = (key) => {
                        const idx = textLines.findIndex(l => l.trim().includes(key));
                        if (idx !== -1 && textLines[idx+1]) {
                            const val = parseFloat(textLines[idx+1].replace(/[^0-9.]/g, ''));
                            return isNaN(val) ? null : val;
                        }
                        return null;
                    };

                    results.PlotValues.Close = findValue('Close');
                    results.PlotValues.MangoD1 = findValue('MangoD1') || findValue('Mango D1');
                    results.PlotValues.MangoD2 = findValue('MangoD2') || findValue('Mango D2');
                    
                    // If Trend is still unknown but we have plots, calculate it
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
                    
                    // Final Visual Fallback: Look for Green/Red keywords if Trend is STILL unknown
                    if (results.Trend === "Unknown") {
                        if (/Bullish|Strong Bull/i.test(bodyText)) results.Trend = "Bullish";
                        else if (/Bearish|Strong Bear/i.test(bodyText)) results.Trend = "Bearish";
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

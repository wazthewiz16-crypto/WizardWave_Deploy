import asyncio
import re
import json
import os
import time
from datetime import datetime
from playwright.async_api import async_playwright

# --- CONFIG ---
STATE_FILE = "tv_state.json"
DATA_FILE = "tv_raw_data.json"

# Full List from app.py
ASSETS = [
    # Crypto (Testing Subset)
    {"symbol": "BINANCE:BTCUSDT.P", "name": "BTC"},
    {"symbol": "BINANCE:ETHUSDT.P", "name": "ETH"},
]

TIMEFRAMES = ["15m", "1h", "4h", "1d", "4d"]

async def scrape_cycle():
    async with async_playwright() as p:
        print(f"[{datetime.now()}] Launching Browser...")
        browser = await p.chromium.launch(headless=True, args=['--no-sandbox'])
        context = await browser.new_context(storage_state=STATE_FILE, viewport={'width': 1920, 'height': 1080})
        page = await context.new_page()

        all_data = []

        try:
            for asset in ASSETS:
                sym = asset['symbol']
                name = asset['name']
                print(f"Processing {name}...")
                
                # Navigate
                url = f"https://www.tradingview.com/chart/?symbol={sym}"
                try:
                    await page.goto(url, timeout=45000)
                    await page.wait_for_timeout(4000)
                    
                    # Ensure Data Window is Open
                    # We press Alt+D once. If it toggles it status, that's a risk.
                    # But assuming fresh session per asset? No, shared session.
                    # If we toggle ON for BTC, it stays ON for ETH.
                    # So we should only press Alt+D for the FIRST asset?
                    # Let's try pressing it every time for robustness? No, that would toggle ON/OFF/ON.
                    # Better: Press it once at the start of the entire cycle? 
                    # Actually, the Data Window state persists in the *tab* or *local storage*.
                    # Let's try pressing it ONLY for the first asset.
                    if asset == ASSETS[0]:
                        print("  > Toggling Data Window (First Run)...")
                        await page.keyboard.press("Alt+d")
                        await page.wait_for_timeout(1000)

                    asset_results = {}
                    
                    for tf in TIMEFRAMES:
                        # Switch TF
                        await page.keyboard.type(tf)
                        await page.keyboard.press("Enter")
                        await page.wait_for_timeout(2500) # Fast switch
                        
                        # Data Window needs a hover
                        await page.mouse.move(1400, 500)
                        await page.wait_for_timeout(300)
                        
                        # Scrape Text
                        content = await page.inner_text("body")
                        
                        # Parse
                        data = parse_content(content)
                        asset_results[tf] = data
                        # print(f"  {tf}: C={data.get('close')}")
                    
                    all_data.append({
                        "asset": name,
                        "raw_data": asset_results,
                        "scraped_at": datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    print(f"  FAILED {name}: {e}")
                    # Continue to next asset
            
            # Atomic Write
            print(f"Writing data for {len(all_data)} assets...")
            tmp = f"{DATA_FILE}.tmp"
            with open(tmp, "w") as f:
                json.dump(all_data, f, indent=4)
            os.replace(tmp, DATA_FILE)
            print("Cycle Complete.")

        finally:
            await browser.close()

def parse_content(text):
    # Reuse valid regex logic
    def extract_val(label, txt):
        pattern = re.compile(re.escape(label) + r"[\s\n]+([0-9,.]+)", re.IGNORECASE)
        match = pattern.search(txt)
        if match:
             try: return float(match.group(1).replace(',', ''))
             except: return None
        return None

    close = extract_val("C", text)
    
    # Parse Mango Block
    # Need to be very robust here.
    mango_vals = []
    
    # Text dump structure based on user image:
    # Mango Dynamic V5
    # MangoD1 1.9724
    # MangoD2 2.0425
    # BuyOp 1.9218
    # Entry Zone Upper 2.0750
    # Entry Zone Lower 1.9624
    # ...
    # Mango Equilibrium Tracker
    
    # Find "Mango Dynamic V5"
    idx = text.find("Mango Dynamic V5")
    if idx != -1:
        sub = text[idx+16:] # Skip title
        lines = [l.strip() for l in sub.split('\n') if l.strip()]
        
        for line in lines:
             # STOP if we hit the next indicator
             if "Equilibrium" in line or "Delta" in line or "Vol" in line:
                 break
             
             # Extract Number from line (it might be "Label Value" or just "Value")
             # Regex to find the LAST number in the line (Value is usually on the right)
             matches = re.findall(r'[+\-0-9,.]+%?', line)
             if matches:
                 # Take the last match as the value
                 val_str = matches[-1].replace(',', '').replace('%', '')
                 try:
                     v = float(val_str)
                     # Sanity check: Value must be > 0
                     if v > 0:
                        mango_vals.append(v)
                 except: pass
             
             if len(mango_vals) >= 5: break
    
    return {
        "close": close,
        "mango": mango_vals
    }

if __name__ == "__main__":
    import sys
    run_once = "--once" in sys.argv
    
    if run_once:
        print("Running Single Cycle...")
        asyncio.run(scrape_cycle())
    else:
        while True:
            try:
                asyncio.run(scrape_cycle())
            except Exception as e:
                print(f"Main Loop Crash: {e}")
                
            print("Sleeping 15m...")
            time.sleep(900) # 15 min loop

import asyncio
import os
from playwright.async_api import async_playwright

async def run():
    print("\n" + "="*50)
    print("--- TRADINGVIEW SESSION SETUP ---")
    print("Launching headful browser...")
    print("1. A browser window will open shortly.")
    print("2. Please log in to your TradingView account.")
    print("3. Navigate to your 'Arcane Portal' layout.")
    print("4. The script will wait for 3 minutes to give you time.")
    print("5. After 3 minutes, it will automatically save the state and close.")
    print("="*50 + "\n")

    async with async_playwright() as p:
        # Launch headful browser
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()

        # Open TradingView
        page = await context.new_page()
        await page.goto("https://www.tradingview.com/chart/qR1XTue9/")

        # Wait for 3 minutes (180 seconds)
        print("Waiting 180 seconds for you to log in and set up the chart...")
        for i in range(180, 0, -10):
            print(f"Time remaining: {i}s...")
            await asyncio.sleep(10)

        # Save storage state
        await context.storage_state(path="tv_state.json")
        print("\n[SUCCESS] Login state saved to tv_state.json")

        await browser.close()

if __name__ == "__main__":
    if not os.path.exists("market_data_cache"):
        os.makedirs("market_data_cache")
    asyncio.run(run())

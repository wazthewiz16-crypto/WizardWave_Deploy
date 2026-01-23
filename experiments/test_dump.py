import asyncio
import os
from playwright.async_api import async_playwright

async def run():
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=True)
    context = await browser.new_context(
        storage_state='tv_state.json',
        viewport={'width': 1920, 'height': 1080}
    )
    page = await context.new_page()
    print("Navigating to TradingView...")
    await page.goto('https://www.tradingview.com/chart/qR1XTue9/?symbol=BINANCE:BTCUSDT')
    
    # Wait for the chart to load
    print("Waiting 15s for chart and indicators...")
    await asyncio.sleep(15)
    
    # Click Data Window if needed (it's usually a button on the right)
    # The subagent clicked pixel (986, 203) - let's try to find it by selector if possible or just use evaluation
    
    print("Extracting DOM text...")
    content = await page.evaluate("() => document.body.innerText")
    
    with open('tv_dump.txt', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Extraction complete. Check tv_dump.txt")
    await browser.close()
    await p.stop()

if __name__ == "__main__":
    asyncio.run(run())

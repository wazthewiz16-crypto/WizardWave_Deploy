---
name: TradingView Playwright Automation
description: Expert system for automating TradingView interactions and scraping indicator data using Playwright. Covers setup, reliable hotkey interactions (Alt+D), data extraction, and cloud deployment best practices.
---

# TradingView Playwright Automation Skill

This skill provides expert knowledge and code patterns for scraping TradingView charts using Playwright. It focuses on reliability in headless cloud environments (Streamlit Cloud, Docker) and extracting custom indicator data.

## 1. Environment Setup

### Installation & Self-Healing
Streamlit Cloud and other ephemeral environments may lose dependencies. Always include self-healing checks at the start of your script.

```python
import subprocess
import sys
import os

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("[!] Playwright module missing. Installing at runtime...")
    # Install with essential dependencies
    subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright==1.49.0", "greenlet", "pyee", "typing-extensions"])
    
    # Reload import system
    import importlib
    import site
    importlib.invalidate_caches()
    if hasattr(site, 'getusersitepackages'):
        user_site = site.getusersitepackages()
        if user_site not in sys.path:
            sys.path.append(user_site)
            
    from playwright.async_api import async_playwright

# Ensure Browsers are Installed
try:
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
except Exception as e:
    print(f"[!] Warning: Browser install failed: {e}")
```

### Browser Launch Configuration
Use these arguments for maximum stability in cloud/container environments:

```python
browser = await p.chromium.launch(
    headless=True,
    args=[
        '--disable-dev-shm-usage', # Critical for Docker/Cloud memory limits
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-gpu',
        '--disable-software-rasterizer',
        '--window-size=1280,800' # Minimum size to prevent UI collapse
    ]
)

context = await browser.new_context(
    storage_state="tv_state.json", # Load session cookies
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    viewport={'width': 1280, 'height': 800},
    ignore_https_errors=True
)
```

## 2. Reliable Interactions

TradingView's DOM is heavily obfuscated and reactive. Using keyboard shortcuts is significantly more reliable than CSS selectors.

### Toggling the Data Window
The "Data Window" contains the precise values of all indicators.
**Best Practice**: Use `Alt + D` after focusing the chart.

```python
async def open_data_window(page):
    print("  [>] Toggling Data Window (Alt+D)...")
    try:
        # 1. Click chart center to ensure focus
        await page.mouse.click(640, 400) 
        await asyncio.sleep(0.5)
        
        # 2. Press Hotkey
        await page.keyboard.press("Alt+D")
        await asyncio.sleep(1.5) # Wait for animation
    except Exception as e:
        print(f"  [!] Failed to open Data Window: {e}")
```

### Changing Timeframes
Typing the timeframe code and pressing Enter is universal.

```python
async def switch_timeframe(page, tf):
    # tf examples: "1h", "4h", "1d", "15" (minutes)
    print(f"  [-] Switching to {tf}...")
    await page.keyboard.type(tf)
    await page.keyboard.press("Enter")
    await asyncio.sleep(3) # Wait for chart reload
```

## 3. Data Extraction Strategy

Do not rely on CSS class names (e.g., `.tv-data-window-item`) as they change frequently. Instead, parse the text content of the page or the Data Window container.

### Parsing Logic (JavaScript Evaluation)

```python
data = await page.evaluate("""() => {
    const results = {
        Trend: "Unknown",
        Values: {}
    };

    // 1. Get all text lines logic
    const textLines = document.body.innerText.split('\\n');
    
    // 2. Find Indicator Text
    // Example: "Trend: Bullish" in Data Window
    const trendLine = textLines.find(l => l.includes('Trend:'));
    if (trendLine) {
         results.Trend = trendLine.split(':')[1].trim();
    }
    
    // 3. Fallback: Search explicitly for keys
    if (results.Trend === "Unknown") {
        if (document.body.innerText.includes("Bullish")) results.Trend = "Bullish";
        if (document.body.innerText.includes("Bearish")) results.Trend = "Bearish";
    }
    
    return results;
}""")
```

## 4. Debugging & Maintenance

*   **Screenshots**: Always take screenshots on exception or if "No Data" is found.
    ```python
    await page.screenshot(path=f"debug_error_{asset_name}_{timestamp}.png")
    ```
*   **Logs**: Write to a file (`scraper_debug.log`) for inspection in headless environments.
*   **Session State**: If the chart looks "default" (no custom indicators), your `tv_state.json` (cookies) is likely invalid or expired. You must re-authenticate and save the state.

## 5. Common Issues & Fixes

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| **"ModuleNotFoundError: playwright"** | Cloud env cleared packages. | Use the runtime install snippet above. |
| **Elements Not Found** | Layout ID mismatch or session invalid. | Verify `tv_state.json` and checking URL `chart/<layout_id>`. |
| **"Unknown" Data** | Data Window closed. | Use `Alt+D` hotkey logic. Check screenshots. |
| **TimeoutError** | Network lag or heavy JS. | Increase `timeout` in `page.goto()` and use `wait_until='load'`. |

---
name: TV Charting Library Expert
description: Expert system for interacting, scraping, and debugging TradingView Advanced Charts and SuperCharts.
---

# TV Charting Library Expert

This skill provides specialized knowledge for interacting with TradingView's DOM, handling charting libraries, and scraping indicator data from the "Data Window".

## Core Capabilities
1.  **DOM Interaction**: Identifying and manipulating TradingView's complex, obfuscated DOM structure.
2.  **Data Extraction**: Scraping numerical values from the "Data Window" (the side panel that shows indicator values).
3.  **Headless Browsing**: Optimizing Playwright scripts to run reliably in headless cloud environments (Streamlit Cloud, Docker).

## TradingView DOM Selectors (Reference)
*   **Data Window Button**: `[data-name="data-window"]`
*   **Object Tree**: `[data-name="object-tree"]`
*   **Right Toolbar**: `div[class*="right-toolbar"]` (often contains the Data Window toggle)
*   **Chart Area**: `table.chart-markup-table`

## Best Practices for Scraping
*   **Viewport Size**: TV is reactive. If the width < 1000px, toolbars may collapse. Use at least `1280x800`.
*   **Hotkeys**: Prefer hotkeys over clicks when possible.
    *   `Shift + D`: Toggle Data Window
*   **Wait Strategy**: Always wait for specific elements (like the Legend or Toolbar) before attempting interactions. `page.wait_for_selector()` is essential.

## Troubleshooting "Unknown" Values
If the scraper returns "Unknown" for trends:
1.  **Check Data Window Visibility**: The scraper relies on the Data Window text being in the DOM. If it's closed, data is likely missing.
2.  **Verify Layout ID**: Ensure the `chart_id` in the URL matches a layout that *actually has* the indicators saved.
3.  **Cookies/Session**: TV layouts are private. Without valid `tv_state.json` (cookies), you will see a default chart without indicators.

## Code Snippet: Robust Data Window Opening
```python
# Try Hotkey First (Shift + D)
await page.keyboard.press("Shift+D")
await asyncio.sleep(1)

# Fallback: Click Loop
await page.evaluate('''() => {
    const btn = document.querySelector('[data-name="data-window"]');
    if (btn) btn.click();
}''')
```

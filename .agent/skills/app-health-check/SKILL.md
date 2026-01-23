---
name: System Doctor
description: Ensure the Streamlit app and data feeds are healthy before the trading week starts.
---

# System Doctor Skill

This skill performs a comprehensive system audit to ensure all trading infrastructure is ready for the high-intensity trading week.

## Usage

### How to Run
```bash
python .agent/skills/app-health-check/scripts/system_audit.py
```

## Logic Description

1. **API Latency Test**:
   - Pings Exchange APIs (Bybit, Binance, etc.) via `CCXT`.
   - Alert: If latency > 500ms, flag as `"Connection Unstable"`.
2. **Data Integrity Check**:
   - Scans the CSV/database files for the latest candles.
   - Detects `NaN` values, price spikes (outliers), or "Gaps" (missing time intervals).
3. **Log Audit**:
   - Opens `app.log`.
   - Uses regex to find `"CRITICAL"`, `"ERROR"`, or `"Traceback"` in the last 48 hours.
4. **Environment Check**:
   - Verifies all required dependencies in `requirements.txt` are installed.
   - Checks disk space and memory usage for the Streamlit server.

## Output
- **System Healthy (Green)**: All tests passed.
- **Maintenance Required (Red)**: Specific errors listed for immediate CTO attention.

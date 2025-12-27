---
description: Start the Background Signal Monitor
---
This workflow starts the standalone signal monitoring script. This script runs indefinitely, checking for new trade signals and sending Discord alerts even if the Streamlit app is closed or snoozing.

1. Open a new terminal instance (so it runs in the background).
2. Run the monitor script:
   ```powershell
   python monitor_signals.py
   ```
3. Keep this terminal window open to ensure alerts continue 24/7.

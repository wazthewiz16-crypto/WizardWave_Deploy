import json
import os
import pandas as pd

TRADES_FILE = "my_trades.json"

def load_trades():
    if not os.path.exists(TRADES_FILE):
        return []
    try:
        with open(TRADES_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_trades_list(trades):
    with open(TRADES_FILE, 'w') as f:
        json.dump(trades, f, indent=2)

def toggle_trade(trade_row, is_saved):
    """
    Adds or removes a trade from the stored list.
    We use (Asset, Time) as a unique composite key.
    """
    trades = load_trades()
    
    # Define Key
    # Sort Key is best, but if it's not present in UI row, we use Time string.
    # trade_row is a dict from the dataframe row.
    
    # We need to robustly identify the trade.
    # Assuming 'Asset' and 'Time' are present and unique enough.
    # Ideally '_sort_key' should be preserved.
    
    target_asset = trade_row.get('Asset')
    target_time = trade_row.get('Time')
    
    if not target_asset or not target_time:
        return # Cannot identify
    
    # Check existence
    existing_idx = -1
    for i, t in enumerate(trades):
        if t.get('Asset') == target_asset and t.get('Time') == target_time:
            existing_idx = i
            break
            
    if is_saved:
        # User wants it saved.
        if existing_idx == -1:
            # Add it
            # We want to store the full trade details
            trades.append(trade_row)
            save_trades_list(trades)
    else:
        # User wants it removed (unchecked)
        if existing_idx != -1:
            trades.pop(existing_idx)
            save_trades_list(trades)

def get_saved_keys():
    """Returns a set of (Asset, Time) tuples for currently saved trades."""
    trades = load_trades()
    return set((t.get('Asset'), t.get('Time')) for t in trades)

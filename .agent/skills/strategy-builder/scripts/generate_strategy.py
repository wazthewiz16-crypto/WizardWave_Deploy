import argparse
import sys
import os
import re

TEMPLATE = r'''
import pandas as pd
import pandas_ta as ta

class {ClassName}:
    """
    {Description}
    """
    def __init__(self, lookback=14):
        self.lookback = lookback
        
    def apply(self, df):
        """
        Applies strategy logic to DataFrame.
        Expected columns: open, high, low, close, volume
        Modifies df in-place or returns new df with 'signal_type' column.
        """
        df = df.copy()
        
        # --- Indicators ---
        # Example: df['rsi'] = ta.rsi(df['close'], length=self.lookback)
        {IndicatorLogic}
        
        # --- Signal Logic ---
        df['signal_type'] = 'NONE'
        
        # Vectorized Logic (Preferred) or Iteration
        {SignalLogic}
        
        return df
'''

def generate_code(name, description):
    # This is a stub. In a real LLM-powered agent, this would use the LLM to generate code.
    # For this deterministic script, we produce a scaffold.
    
    class_name = "".join(x.title() for x in name.split('_')) + "Strategy"
    
    # Simple heuristic to fill logic (User would usually edit this)
    indicator_logic = "# TODO: Add indicators based on: " + description
    signal_logic = "# TODO: Implement logic for: " + description
    
    return TEMPLATE.format(
        ClassName=class_name,
        Description=description,
        IndicatorLogic=indicator_logic,
        SignalLogic=signal_logic
    )

def main():
    if len(sys.argv) < 3:
        print("Usage: generate_strategy.py <StrategyName> <Description>")
        sys.exit(1)
        
    name = sys.argv[1]
    desc = sys.argv[2]
    
    # 1. Generate Code
    code = generate_code(name, desc)
    
    # 2. Save File
    filename = f"strategy_{name.lower()}.py"
    # Save to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    filepath = os.path.join(project_root, filename)
    
    with open(filepath, 'w') as f:
        f.write(code)
    
    print(f"[+] Created {filename} in project root.")
    print(f"    Class: {filename.replace('strategy_', '').replace('.py', '').title().replace('_','')}Strategy")
    print("    Action: Edit the file to implement specific logic.")

if __name__ == "__main__":
    main()

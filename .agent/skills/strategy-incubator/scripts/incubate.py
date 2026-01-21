import os
import subprocess
import shutil
import argparse
import sys

def run_command(command):
    print(f"Running: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error: {stderr}")
        return False, stdout, stderr
    return True, stdout, stderr

def incubate_experiment(name):
    print(f"--- Launching Strategy Incubator: {name} ---")
    
    # 1. Check Git Status
    success, stdout, stderr = run_command("git status --porcelain")
    if stdout.strip():
        print("[!] Warning: Git working directory is not clean. It is recommended to commit or stash changes first.")
    
    # 2. Create Branch
    branch_name = f"feature/{name}"
    print(f"Creating branch: {branch_name}")
    success, _, _ = run_command(f"git checkout -b {branch_name}")
    if not success:
        print("Branch already exists or error occurred. Switching...")
        run_command(f"git checkout {branch_name}")

    # 3. Fork Config
    if os.path.exists("strategy_config.json"):
        print("Forking strategy_config.json -> strategy_config_experimental.json")
        shutil.copy("strategy_config.json", "strategy_config_experimental.json")
    
    # 4. Suggest Next Steps
    print("\nDONE: Sandbox Environment Ready!")
    print(f"Current Branch: {branch_name}")
    print("Config: strategy_config_experimental.json")
    print("-" * 50)
    print("Action Required: Modify strategy_config_experimental.json with your crypto CLS tweaks.")
    print("To launch the experimental dashboard, run:")
    print("streamlit run app.py --server.port 8502")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Name of the experiment")
    args = parser.parse_args()
    
    incubate_experiment(args.name)

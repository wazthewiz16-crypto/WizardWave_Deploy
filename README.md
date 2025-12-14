# Wizard Wave V9 - ML Dashboard Deployment

## How to Host 24/7 on Streamlit Cloud

This application is ready to be hosted on **Streamlit Cloud** (Free Community Tier).

### Step 1: Push to GitHub
1.  Create a **New Repository** on GitHub (e.g., `wizard-wave-signals`).
2.  Open a terminal in this folder (`WizardWave_Deploy`).
3.  Run the following commands:
    ```bash
    git init
    git add .
    git commit -m "Initial deploy"
    git branch -M main
    git remote add origin https://github.com/YOUR_USERNAME/wizard-wave-signals.git
    git push -u origin main
    ```

### Step 2: Deploy on Streamlit
1.  Go to [share.streamlit.io](https://share.streamlit.io/).
2.  Click **"New app"**.
3.  Select your GitHub repository (`wizard-wave-signals`).
4.  Set **Main file path** to `app.py`.
5.  Click **"Deploy"**.

### Notes
- **Model File**: The `model.pkl` is included. Streamlit Cloud has a limit of ~1GB, so typical models work fine.
- **Dependencies**: The `requirements.txt` file ensures all libraries (`pandas`, `pandas_ta`, `sklearn`, `ccxt`, `yfinance`) are installed automatically.
- **24/7 Uptime**: Streamlit Cloud will keep the app running. If it goes to sleep after inactivity, simply visiting the URL wakes it up.

@echo off
title Oracle Service Launcher
echo ===================================================
echo   🔮 Starting WizardWave Oracle Services (V2)
echo ===================================================
echo.

echo [1/2] Launching TradingView Scraper...
start "Oracle Scraper" cmd /k "python scrape_tv.py"

echo [2/2] Launching Alert Manager...
start "Oracle Alerter" cmd /k "python alert_manager.py"

echo.
echo Services Launched! 
echo You can minimize this window, or close it (services stay running).
echo.
pause

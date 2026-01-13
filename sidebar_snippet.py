
# --- TRADING PLAN SIDEBAR ---
with st.sidebar:
    st.markdown("### 📜 My Trading Plan")
    st.info("Define your rules to streamline your process.")
    
    # 1. Filters
    st.caption("Auto-Filtering")
    plan_active = st.checkbox("Apply Plan Filters", value=True, help="Automatically filter Runic Alerts & History based on rules.")
    
    min_conf = st.slider("Min Confidence", 0, 100, 55, step=5)
    
    allowed_tfs = st.multiselect(
        "Allowed Timeframes", 
        ["1H", "4H", "12H", "1D", "4D"],
        default=["4H", "1D"],
        help="Select timeframes you want to focus on."
    )
    
    # 2. Daily Limits
    st.caption("Risk Management")
    max_daily_trades = st.number_input("Max Trades Per Day", 1, 10, 2)
    
    # Count Today's Trades (from Manage Trades)
    try:
        import manage_trades, datetime
        curr_saved_trades = manage_trades.load_trades()
        today_str = datetime.date.today().strftime('%Y-%m-%d')
        # Assuming trade['Time'] has date, parse it. 
        # But trade['Time'] format is "YYYY-MM-DD HH:MM..."
        todays_count = sum(1 for t in curr_saved_trades if str(t.get('Time')).startswith(today_str))
        
        st.write(f"**Trades Taken Today:** {todays_count} / {max_daily_trades}")
        if todays_count >= max_daily_trades:
            st.error("🚫 Daily Limit Reached! Stop Trading.")
            st.toast("Daily Limit Reached! Step away from the charts.", icon="🛑")
        else:
            p = todays_count / max_daily_trades
            st.progress(p)
            st.caption(f"{max_daily_trades - todays_count} trades remaining.")
            
    except Exception as e:
        st.error(f"Error checking daily limit: {e}")

    # Store in Session State
    st.session_state['plan_active'] = plan_active
    st.session_state['plan_min_conf'] = min_conf
    st.session_state['plan_allowed_tfs'] = allowed_tfs


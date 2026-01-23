# Troubleshooting Streamlit Cloud "Access Denied"

If you see the error **"You do not have access to this app or it does not exist"**, follow these steps:

## 1. Check Repository Visibility
If your GitHub repository is **Private**, Streamlit Cloud needs explicit permission to see it.
- **Quick Fix**: Change your repository visibility to **Public** in GitHub Settings -> General -> Danger Zone.
- **Private Fix**: If you must keep it private, go to your Streamlit Cloud Dashboard, click "New App", and ensure you grant Streamlit access to your repositories when prompted.

## 2. Re-Connect GitHub
Sometimes the link between Streamlit and GitHub gets stale.
1. Sign out of Streamlit Cloud.
2. Sign back in using **"Continue with GitHub"**.
3. Go to Streamlit Settings -> Linked Accounts -> Revoke GitHub, then re-link it.

## 3. Verify Account Match
Ensure you are deploying the app with the **same GitHub account** that owns the repository.
- The error message showed you are signed in as `wazthewiz16`.
- Ensure the repo is under `wazthewiz16-crypto` (or whichever GitHub user owns it).

## 4. Redeploy
If the app failed to build initially, it might show this error.
1. Delete the app from your Streamlit dashboard.
2. Click **"New App"**.
3. Select your repo again and deploy.

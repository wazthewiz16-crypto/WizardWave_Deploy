---
name: Strategy Incubator
description: safely create a new Git branch and environment to test wild ideas without breaking the main app.
---

# Strategy Incubator Skill

This skill allows for safe experimentation by setting up a "Sandbox" environment. It handles version control and environment isolation for new strategy ideas.

## Usage

### Parameters
- **Experiment Name**: (e.g., `mean-reversion-test`)

### How to Run
```bash
# Call from the agent with the experiment name
.agent/skills/strategy-incubator/scripts/incubate.py --name "mean-reversion-test"
```

## Logic Description

1. **Working Directory Check**: Runs `git status` to ensure there are no uncommitted changes that could cause conflicts.
2. **Branch Isolation**:
   - Creates a new Git branch: `feature/[experiment-name]`.
   - Switches to the branch immediately.
3. **Configuration Forking**:
   - Copies `config.json` (or your main config) to `config_experimental.json`.
   - Allows the user to modify experimental parameters without touching the production-ready config.
4. **Sandbox Deployment**:
   - Launches a separate Streamlit instance on a different port (e.g., `8502`).
   - This allows you to view the "Wild Idea" dashboard side-by-side with the "Stable" version on `8501`.

## Output
Confirmation of the new branch and the URL/Port for the experimental Streamlit instance.

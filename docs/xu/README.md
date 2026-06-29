# Xu Search-R1 Experiment Notes

This directory archives the local experiment notes used while reproducing and extending Search-R1 on the NUIST A100 server.

Contents:

- `deepseek_one_week_readme.md`: working README for environment, paths, SSH routes, Docker images, W&B, and experiment conventions.
- `search_r1_sync_log.md`: chronological experiment and synchronization log, including smoke tests, PPO/GRPO runs, evaluation results, and known issues.
- `../../figures/xu/`: generated metric plots and CSV files used for PPO/GRPO analysis.

Large runtime artifacts are intentionally excluded from Git:

- A100 logs, Ray logs, W&B run directories
- model weights, datasets, retrieval indexes
- checkpoints and generated cache directories
- `.env`, `.secrets`, API keys, and machine-local credentials

# Experiment Artifacts

This directory keeps only the current paper-facing experiment snapshot:

- `stage7_llm/`: semantic intent library condition, kept for the current Stage7 main result.
- `stage7_baseline_onehot/`: one-hot intent baseline for Stage7.
- `vmas_stage1/`: VMAS dispersion sanity/limitation result.
- `STAGE7_REPORT.md` and `stage7_results.json`: compact summary of the main Stage7 comparison.

Do not commit training checkpoints or high-frequency intermediate logs here. Keep only:

- `summary.json`
- per-seed `result.json`
- final comparison/training figures
- intent library files needed to reproduce the condition


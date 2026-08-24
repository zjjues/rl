# Superseded two-run resume fragment: derived-objective protocol drift

The preserved directory `experiments/paper/uav_imappo_ablation_paper_v2_superseded_resume_objective_drift_20260824/` contains valid numerical runs for `imappo_full × seeds {7, 11}` with one shared implementation SHA-256. It must not enter paper statistics.

The first cross-commit `--resume` call exposed a canonical-protocol bug: `merge_resume_specs` converted an unchanged scalar `objective` into the same scalar plus a redundant one-element `objectives` array. The execution fields were unchanged, but the full registered spec bytes and therefore the study/result protocol SHA-256 changed. Strict partial validation correctly rejected seed 11 and the manifest instead of treating descriptive-field drift as equivalent.

The fragment is preserved unchanged with its invalid audit at `docs/paper/audits/uav_imappo_ablation_resume_objective_drift_superseded_audit.json`. The runner now keeps identical single-objective resume specs byte-for-byte idempotent, with a regression test asserting both dictionary equality and canonical protocol-hash equality. Both registered seeds must be rerun from the corrected clean snapshot.

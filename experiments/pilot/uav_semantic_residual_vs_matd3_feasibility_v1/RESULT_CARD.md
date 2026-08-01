# Cross-study Result Card: uav_semantic_residual_vs_matd3_feasibility_v1

- Contrast: `objective_semantic_residual` minus `matd3`
- Paired seeds: `5`
- Source checksums: verified before comparison
- Pairing basis: identical seeds, environment, risk tiers, evaluation episodes, and query suite

## Main-risk-tier paired differences

| Tier | Metric | Mean difference | 95% bootstrap CI | Treatment win rate |
|---|---:|---:|---:|---:|
| easy | collision_rate | -0.1760 | [-0.3200, -0.0320] | 0.60 |
| easy | task_completion | -0.0586 | [-0.0859, -0.0440] | 0.00 |
| easy | episode_return | 7.7806 | [4.9240, 12.0831] | 1.00 |
| easy | episode_collisions | -3.5200 | [-6.4000, -0.6400] | 0.60 |
| hard | collision_rate | -0.3360 | [-0.5120, -0.2240] | 1.00 |
| hard | task_completion | -0.0847 | [-0.1154, -0.0651] | 0.00 |
| hard | episode_return | 2.3924 | [1.2180, 3.6199] | 1.00 |
| hard | episode_collisions | -6.7200 | [-9.9610, -4.4800] | 1.00 |

## Interpretation guardrails

- Differences are treatment minus baseline; lower is better only for collision metrics.
- Five-seed feasibility intervals are diagnostic, not paper-grade confirmatory evidence.
- A confidence interval containing zero does not support a stable directional claim.
- Generalization comparisons average queries within split and seed before bootstrapping.
- This artifact does not alter either checksummed source study.

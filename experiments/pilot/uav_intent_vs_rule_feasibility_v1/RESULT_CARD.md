# Cross-study Result Card: uav_intent_vs_rule_feasibility_v1

- Contrast: `objective_grounded_semantic` minus `rule_planner`
- Paired seeds: `5`
- Source checksums: verified before comparison
- Pairing basis: identical seeds, environment, risk tiers, evaluation episodes, and query suite

## Main-risk-tier paired differences

| Tier | Metric | Mean difference | 95% bootstrap CI | Treatment win rate |
|---|---:|---:|---:|---:|
| easy | collision_rate | -0.0360 | [-0.1280, 0.1160] | 0.80 |
| easy | task_completion | -0.1345 | [-0.1501, -0.1155] | 0.00 |
| easy | episode_return | -12.9138 | [-17.1848, -9.4503] | 0.00 |
| easy | episode_collisions | -0.7200 | [-2.5600, 2.3200] | 0.80 |
| hard | collision_rate | 0.1400 | [0.0760, 0.2040] | 0.00 |
| hard | task_completion | -0.0742 | [-0.0852, -0.0619] | 0.00 |
| hard | episode_return | -0.6210 | [-2.7190, 2.0995] | 0.20 |
| hard | episode_collisions | 2.8000 | [1.5200, 4.0800] | 0.00 |

## Interpretation guardrails

- Differences are treatment minus baseline; lower is better only for collision metrics.
- Five-seed feasibility intervals are diagnostic, not paper-grade confirmatory evidence.
- A confidence interval containing zero does not support a stable directional claim.
- Generalization comparisons average queries within split and seed before bootstrapping.
- This artifact does not alter either checksummed source study.

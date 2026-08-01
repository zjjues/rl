# Intent Controllability Card: uav_semantic_residual_controllability_exploratory_v1

- Evidence status: exploratory post-hoc diagnostic; pre-register before paper use
- Source checksum audit: `verified`
- Treatment: `objective_semantic_residual`
- Statistical unit: one trained seed; queries are correlated within-seed probes

## All-query safety-tradeoff alignment

| Variant | Tier | Mean Spearman | 95% bootstrap CI |
|---|---|---:|---:|
| objective_semantic_residual | easy | 0.182 | [-0.008, 0.401] |
| objective_semantic_residual | hard | 0.351 | [0.158, 0.553] |
| raw_semantic_residual | easy | 0.018 | [-0.115, 0.240] |
| raw_semantic_residual | hard | 0.169 | [-0.040, 0.372] |
| nonsemantic_residual | easy | -0.104 | [-0.451, 0.169] |
| nonsemantic_residual | hard | 0.220 | [-0.065, 0.456] |
| rule_planner_oracle | easy | 0.571 | [0.443, 0.700] |
| rule_planner_oracle | hard | 0.667 | [0.556, 0.775] |

A positive correlation means intents that request more safety relative to task speed move behavior toward fewer collisions and/or lower completion. Correlation alone does not establish superiority; absolute safety and completion remain co-primary outcomes.

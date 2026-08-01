# PyBullet Transfer Result Card: pybullet_transfer_smoke_v1

- Evidence level: `smoke`
- Paired seeds: `3`
- Simulator: headless Crazyflie rigid-body/rotor dynamics through VelocityAviary
- Scope: high-level controller and safety-layer transfer; this is not SITL, HIL, or real flight

## Treatment summary

| Metric | Mean | 95% bootstrap CI |
|---|---:|---:|
| minimum_pairwise_distance | 0.2088 | [0.2061, 0.2118] |
| collision_step_fraction | 0.0000 | [0.0000, 0.0000] |
| goal_success_fraction | 0.0000 | [0.0000, 0.0000] |
| final_goal_rmse | 0.5442 | [0.5246, 0.5684] |
| normalized_command_energy | 0.2216 | [0.2186, 0.2233] |
| mean_filter_correction | 0.0893 | [0.0850, 0.0935] |
| solver_success_fraction | 1.0000 | [1.0000, 1.0000] |
| constraint_max_violation | 0.0000 | [0.0000, 0.0000] |
| mean_solver_time_ms | 0.7556 | [0.3651, 1.5356] |
| safety_distance | 0.2347 | [0.2347, 0.2347] |
| speed_limit_mps | 0.2500 | [0.2500, 0.2500] |

## Paired treatment differences

Differences are treatment minus baseline.

| Baseline | Metric | Mean difference | 95% bootstrap CI |
|---|---|---:|---:|
| cyclic_projection | minimum_pairwise_distance | 0.0006 | [-0.0018, 0.0037] |
| cyclic_projection | collision_step_fraction | 0.0000 | [0.0000, 0.0000] |
| cyclic_projection | goal_success_fraction | 0.0000 | [0.0000, 0.0000] |
| cyclic_projection | final_goal_rmse | 0.0013 | [-0.0092, 0.0112] |
| cyclic_projection | normalized_command_energy | -0.0002 | [-0.0016, 0.0011] |
| cyclic_projection | mean_filter_correction | 0.0022 | [-0.0004, 0.0036] |
| cyclic_projection | solver_success_fraction | 0.0000 | [0.0000, 0.0000] |
| cyclic_projection | constraint_max_violation | 0.0000 | [0.0000, 0.0000] |
| cyclic_projection | mean_solver_time_ms | -0.0264 | [-0.4058, 0.7208] |
| cyclic_projection | safety_distance | 0.0000 | [0.0000, 0.0000] |
| cyclic_projection | speed_limit_mps | 0.0000 | [0.0000, 0.0000] |
| no_filter | minimum_pairwise_distance | 0.0864 | [0.0817, 0.0917] |
| no_filter | collision_step_fraction | 0.0000 | [0.0000, 0.0000] |
| no_filter | goal_success_fraction | -0.0278 | [-0.0833, 0.0000] |
| no_filter | final_goal_rmse | 0.0581 | [0.0352, 0.0734] |
| no_filter | normalized_command_energy | -0.0225 | [-0.0233, -0.0221] |
| no_filter | mean_filter_correction | 0.0893 | [0.0850, 0.0935] |
| no_filter | solver_success_fraction | 0.6693 | [0.6567, 0.6778] |
| no_filter | constraint_max_violation | -0.1894 | [-0.1933, -0.1829] |
| no_filter | mean_solver_time_ms | 0.7556 | [0.3651, 1.5356] |
| no_filter | safety_distance | 0.0000 | [0.0000, 0.0000] |
| no_filter | speed_limit_mps | 0.0000 | [0.0000, 0.0000] |

## Interpretation guardrails

- Smoke and five-seed pilot intervals are diagnostic, not confirmatory.
- A rigid-body simulation improves dynamics validity but does not establish sim-to-real safety.
- The evaluated controller uses structured objective profiles; independent-language evidence is separate.
- Collision distance and linearized command-space constraints are reported separately.

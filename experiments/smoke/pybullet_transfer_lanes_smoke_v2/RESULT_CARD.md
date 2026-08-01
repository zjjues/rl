# PyBullet Transfer Result Card: pybullet_transfer_lanes_smoke_v2

- Evidence level: `smoke`
- Paired seeds: `3`
- Simulator: headless Crazyflie rigid-body/rotor dynamics through VelocityAviary
- Scope: high-level controller and safety-layer transfer; this is not SITL, HIL, or real flight

## Treatment summary

| Metric | Mean | 95% bootstrap CI |
|---|---:|---:|
| minimum_pairwise_distance | 0.2125 | [0.2091, 0.2141] |
| collision_step_fraction | 0.0000 | [0.0000, 0.0000] |
| goal_success_fraction | 0.7500 | [0.7083, 0.7917] |
| final_goal_rmse | 0.1006 | [0.0962, 0.1064] |
| normalized_command_energy | 0.2325 | [0.2316, 0.2334] |
| mean_filter_correction | 0.0541 | [0.0523, 0.0570] |
| solver_success_fraction | 1.0000 | [1.0000, 1.0000] |
| constraint_max_violation | 0.0000 | [0.0000, 0.0000] |
| mean_solver_time_ms | 0.3726 | [0.3218, 0.4552] |
| safety_distance | 0.2347 | [0.2347, 0.2347] |
| speed_limit_mps | 0.2500 | [0.2500, 0.2500] |

## Paired treatment differences

Differences are treatment minus baseline.

| Baseline | Metric | Mean difference | 95% bootstrap CI |
|---|---|---:|---:|
| cyclic_projection | minimum_pairwise_distance | -0.0000 | [-0.0011, 0.0019] |
| cyclic_projection | collision_step_fraction | 0.0000 | [0.0000, 0.0000] |
| cyclic_projection | goal_success_fraction | -0.0278 | [-0.0833, 0.0417] |
| cyclic_projection | final_goal_rmse | -0.0023 | [-0.0128, 0.0050] |
| cyclic_projection | normalized_command_energy | -0.0001 | [-0.0006, 0.0009] |
| cyclic_projection | mean_filter_correction | -0.0006 | [-0.0016, 0.0001] |
| cyclic_projection | solver_success_fraction | 0.0000 | [0.0000, 0.0000] |
| cyclic_projection | constraint_max_violation | 0.0000 | [0.0000, 0.0000] |
| cyclic_projection | mean_solver_time_ms | -0.3125 | [-0.3698, -0.2236] |
| cyclic_projection | safety_distance | 0.0000 | [0.0000, 0.0000] |
| cyclic_projection | speed_limit_mps | 0.0000 | [0.0000, 0.0000] |
| no_filter | minimum_pairwise_distance | 0.0818 | [0.0796, 0.0830] |
| no_filter | collision_step_fraction | 0.0000 | [0.0000, 0.0000] |
| no_filter | goal_success_fraction | -0.0417 | [-0.0833, 0.0000] |
| no_filter | final_goal_rmse | -0.0027 | [-0.0143, 0.0032] |
| no_filter | normalized_command_energy | -0.0060 | [-0.0092, -0.0032] |
| no_filter | mean_filter_correction | 0.0541 | [0.0523, 0.0570] |
| no_filter | solver_success_fraction | 0.5678 | [0.5479, 0.5931] |
| no_filter | constraint_max_violation | -0.1316 | [-0.1362, -0.1290] |
| no_filter | mean_solver_time_ms | 0.3726 | [0.3218, 0.4552] |
| no_filter | safety_distance | 0.0000 | [0.0000, 0.0000] |
| no_filter | speed_limit_mps | 0.0000 | [0.0000, 0.0000] |

## Interpretation guardrails

- Smoke and five-seed pilot intervals are diagnostic, not confirmatory.
- A rigid-body simulation improves dynamics validity but does not establish sim-to-real safety.
- The evaluated controller uses structured objective profiles; independent-language evidence is separate.
- Collision distance and linearized command-space constraints are reported separately.

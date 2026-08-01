# PyBullet Transfer Result Card: pybullet_robust_qp_smoke_v1

- Evidence level: `smoke`
- Paired seeds: `3`
- Simulator: headless Crazyflie rigid-body/rotor dynamics through VelocityAviary
- Scope: high-level controller and safety-layer transfer; this is not SITL, HIL, or real flight

## Treatment summary

| Metric | Mean | 95% bootstrap CI |
|---|---:|---:|
| minimum_pairwise_distance | 0.2563 | [0.2514, 0.2615] |
| collision_step_fraction | 0.0000 | [0.0000, 0.0000] |
| safety_violation_step_fraction | 0.0007 | [0.0007, 0.0007] |
| goal_success_fraction | 0.6250 | [0.6250, 0.6250] |
| final_goal_rmse | 0.1637 | [0.1600, 0.1703] |
| normalized_command_energy | 0.2357 | [0.2347, 0.2375] |
| mean_filter_correction | 0.0442 | [0.0399, 0.0475] |
| solver_success_fraction | 1.0000 | [1.0000, 1.0000] |
| constraint_max_violation | 0.0000 | [0.0000, 0.0000] |
| mean_solver_time_ms | 0.3985 | [0.3632, 0.4592] |
| safety_distance | 0.2347 | [0.2347, 0.2347] |
| constraint_distance | 0.2747 | [0.2747, 0.2747] |
| robust_margin | 0.0400 | [0.0400, 0.0400] |
| speed_limit_mps | 0.2500 | [0.2500, 0.2500] |

## Paired treatment differences

Differences are treatment minus baseline.

| Baseline | Metric | Mean difference | 95% bootstrap CI |
|---|---|---:|---:|
| qp_cbf | minimum_pairwise_distance | 0.0106 | [0.0099, 0.0115] |
| qp_cbf | collision_step_fraction | 0.0000 | [0.0000, 0.0000] |
| qp_cbf | safety_violation_step_fraction | -0.0030 | [-0.0069, 0.0000] |
| qp_cbf | goal_success_fraction | 0.0139 | [0.0000, 0.0417] |
| qp_cbf | final_goal_rmse | -0.0086 | [-0.0112, -0.0039] |
| qp_cbf | normalized_command_energy | -0.0081 | [-0.0086, -0.0074] |
| qp_cbf | mean_filter_correction | 0.0218 | [0.0205, 0.0237] |
| qp_cbf | solver_success_fraction | 0.0000 | [0.0000, 0.0000] |
| qp_cbf | constraint_max_violation | 0.0000 | [0.0000, 0.0000] |
| qp_cbf | mean_solver_time_ms | -0.0166 | [-0.0815, 0.1078] |
| qp_cbf | safety_distance | 0.0000 | [0.0000, 0.0000] |
| qp_cbf | constraint_distance | 0.0400 | [0.0400, 0.0400] |
| qp_cbf | robust_margin | 0.0400 | [0.0400, 0.0400] |
| qp_cbf | speed_limit_mps | 0.0000 | [0.0000, 0.0000] |
| cyclic_projection | minimum_pairwise_distance | 0.0105 | [0.0088, 0.0128] |
| cyclic_projection | collision_step_fraction | 0.0000 | [0.0000, 0.0000] |
| cyclic_projection | safety_violation_step_fraction | -0.0025 | [-0.0056, 0.0000] |
| cyclic_projection | goal_success_fraction | 0.0139 | [-0.0417, 0.0833] |
| cyclic_projection | final_goal_rmse | -0.0152 | [-0.0384, 0.0063] |
| cyclic_projection | normalized_command_energy | -0.0078 | [-0.0088, -0.0070] |
| cyclic_projection | mean_filter_correction | 0.0217 | [0.0213, 0.0223] |
| cyclic_projection | solver_success_fraction | 0.0000 | [0.0000, 0.0000] |
| cyclic_projection | constraint_max_violation | 0.0000 | [0.0000, 0.0000] |
| cyclic_projection | mean_solver_time_ms | -0.3276 | [-0.3624, -0.2721] |
| cyclic_projection | safety_distance | 0.0000 | [0.0000, 0.0000] |
| cyclic_projection | constraint_distance | 0.0400 | [0.0400, 0.0400] |
| cyclic_projection | robust_margin | 0.0400 | [0.0400, 0.0400] |
| cyclic_projection | speed_limit_mps | 0.0000 | [0.0000, 0.0000] |
| no_filter | minimum_pairwise_distance | 0.0243 | [0.0167, 0.0327] |
| no_filter | collision_step_fraction | 0.0000 | [0.0000, 0.0000] |
| no_filter | safety_violation_step_fraction | -0.0250 | [-0.0347, -0.0194] |
| no_filter | goal_success_fraction | -0.0139 | [-0.1250, 0.1667] |
| no_filter | final_goal_rmse | -0.0625 | [-0.1136, 0.0223] |
| no_filter | normalized_command_energy | -0.0121 | [-0.0174, -0.0084] |
| no_filter | mean_filter_correction | 0.0442 | [0.0399, 0.0475] |
| no_filter | solver_success_fraction | 0.1141 | [0.0896, 0.1424] |
| no_filter | constraint_max_violation | -0.0820 | [-0.0877, -0.0754] |
| no_filter | mean_solver_time_ms | 0.3985 | [0.3632, 0.4592] |
| no_filter | safety_distance | 0.0000 | [0.0000, 0.0000] |
| no_filter | constraint_distance | 0.0400 | [0.0400, 0.0400] |
| no_filter | robust_margin | 0.0400 | [0.0400, 0.0400] |
| no_filter | speed_limit_mps | 0.0000 | [0.0000, 0.0000] |

## Interpretation guardrails

- Smoke and five-seed pilot intervals are diagnostic, not confirmatory.
- A rigid-body simulation improves dynamics validity but does not establish sim-to-real safety.
- The evaluated controller uses structured objective profiles; independent-language evidence is separate.
- Collision distance and linearized command-space constraints are reported separately.

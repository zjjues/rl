# PyBullet Transfer Result Card: pybullet_robust_qp_pilot_v1

- Evidence level: `pilot`
- Paired seeds: `5`
- Simulator: headless Crazyflie rigid-body/rotor dynamics through VelocityAviary
- Scope: high-level controller and safety-layer transfer; this is not SITL, HIL, or real flight

## Treatment summary

| Metric | Mean | 95% bootstrap CI |
|---|---:|---:|
| minimum_pairwise_distance | 0.2585 | [0.2543, 0.2621] |
| collision_step_fraction | 0.0000 | [0.0000, 0.0000] |
| safety_violation_step_fraction | 0.0006 | [0.0006, 0.0006] |
| goal_success_fraction | 0.9167 | [0.8917, 0.9417] |
| final_goal_rmse | 0.0815 | [0.0740, 0.0887] |
| normalized_command_energy | 0.2295 | [0.2276, 0.2314] |
| mean_filter_correction | 0.0554 | [0.0507, 0.0601] |
| solver_success_fraction | 1.0000 | [1.0000, 1.0000] |
| constraint_max_violation | 0.0000 | [0.0000, 0.0000] |
| mean_solver_time_ms | 0.4117 | [0.3848, 0.4434] |
| safety_distance | 0.2347 | [0.2347, 0.2347] |
| constraint_distance | 0.2747 | [0.2747, 0.2747] |
| robust_margin | 0.0400 | [0.0400, 0.0400] |
| speed_limit_mps | 0.2500 | [0.2500, 0.2500] |

## Paired treatment differences

Differences are treatment minus baseline.

| Baseline | Metric | Mean difference | 95% bootstrap CI |
|---|---|---:|---:|
| qp_cbf | minimum_pairwise_distance | 0.0117 | [0.0106, 0.0133] |
| qp_cbf | collision_step_fraction | 0.0000 | [0.0000, 0.0000] |
| qp_cbf | safety_violation_step_fraction | -0.0038 | [-0.0061, -0.0014] |
| qp_cbf | goal_success_fraction | 0.0500 | [0.0083, 0.0917] |
| qp_cbf | final_goal_rmse | -0.0091 | [-0.0156, -0.0027] |
| qp_cbf | normalized_command_energy | -0.0135 | [-0.0149, -0.0124] |
| qp_cbf | mean_filter_correction | 0.0298 | [0.0266, 0.0336] |
| qp_cbf | solver_success_fraction | 0.0000 | [0.0000, 0.0000] |
| qp_cbf | constraint_max_violation | 0.0000 | [0.0000, 0.0000] |
| qp_cbf | mean_solver_time_ms | 0.0017 | [-0.1120, 0.0737] |
| qp_cbf | safety_distance | 0.0000 | [0.0000, 0.0000] |
| qp_cbf | constraint_distance | 0.0400 | [0.0400, 0.0400] |
| qp_cbf | robust_margin | 0.0400 | [0.0400, 0.0400] |
| qp_cbf | speed_limit_mps | 0.0000 | [0.0000, 0.0000] |
| cyclic_projection | minimum_pairwise_distance | 0.0118 | [0.0100, 0.0136] |
| cyclic_projection | collision_step_fraction | 0.0000 | [0.0000, 0.0000] |
| cyclic_projection | safety_violation_step_fraction | -0.0033 | [-0.0057, -0.0010] |
| cyclic_projection | goal_success_fraction | 0.0417 | [0.0083, 0.0750] |
| cyclic_projection | final_goal_rmse | -0.0070 | [-0.0124, -0.0012] |
| cyclic_projection | normalized_command_energy | -0.0129 | [-0.0137, -0.0122] |
| cyclic_projection | mean_filter_correction | 0.0293 | [0.0269, 0.0323] |
| cyclic_projection | solver_success_fraction | 0.0000 | [0.0000, 0.0000] |
| cyclic_projection | constraint_max_violation | 0.0000 | [0.0000, 0.0000] |
| cyclic_projection | mean_solver_time_ms | -2.3890 | [-2.7529, -1.8604] |
| cyclic_projection | safety_distance | 0.0000 | [0.0000, 0.0000] |
| cyclic_projection | constraint_distance | 0.0400 | [0.0400, 0.0400] |
| cyclic_projection | robust_margin | 0.0400 | [0.0400, 0.0400] |
| cyclic_projection | speed_limit_mps | 0.0000 | [0.0000, 0.0000] |
| no_filter | minimum_pairwise_distance | 0.0255 | [0.0215, 0.0297] |
| no_filter | collision_step_fraction | 0.0000 | [0.0000, 0.0000] |
| no_filter | safety_violation_step_fraction | -0.0330 | [-0.0379, -0.0286] |
| no_filter | goal_success_fraction | 0.1250 | [0.0583, 0.1750] |
| no_filter | final_goal_rmse | -0.0482 | [-0.0877, -0.0128] |
| no_filter | normalized_command_energy | -0.0213 | [-0.0244, -0.0182] |
| no_filter | mean_filter_correction | 0.0554 | [0.0507, 0.0601] |
| no_filter | solver_success_fraction | 0.1539 | [0.1257, 0.1857] |
| no_filter | constraint_max_violation | -0.0794 | [-0.0843, -0.0753] |
| no_filter | mean_solver_time_ms | 0.4117 | [0.3848, 0.4434] |
| no_filter | safety_distance | 0.0000 | [0.0000, 0.0000] |
| no_filter | constraint_distance | 0.0400 | [0.0400, 0.0400] |
| no_filter | robust_margin | 0.0400 | [0.0400, 0.0400] |
| no_filter | speed_limit_mps | 0.0000 | [0.0000, 0.0000] |

## Interpretation guardrails

- Smoke and five-seed pilot intervals are diagnostic, not confirmatory.
- A rigid-body simulation improves dynamics validity but does not establish sim-to-real safety.
- The evaluated controller uses structured objective profiles; independent-language evidence is separate.
- Collision distance and linearized command-space constraints are reported separately.

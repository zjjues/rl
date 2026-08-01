# PyBullet Transfer Result Card: pybullet_transfer_lanes_pilot_v1

- Evidence level: `pilot`
- Paired seeds: `5`
- Simulator: headless Crazyflie rigid-body/rotor dynamics through VelocityAviary
- Scope: high-level controller and safety-layer transfer; this is not SITL, HIL, or real flight

## Treatment summary

| Metric | Mean | 95% bootstrap CI |
|---|---:|---:|
| minimum_pairwise_distance | 0.2112 | [0.2098, 0.2127] |
| collision_step_fraction | 0.0000 | [0.0000, 0.0000] |
| safety_violation_step_fraction | 0.2383 | [0.2154, 0.2609] |
| goal_success_fraction | 0.9917 | [0.9750, 1.0000] |
| final_goal_rmse | 0.0425 | [0.0392, 0.0460] |
| normalized_command_energy | 0.2267 | [0.2253, 0.2282] |
| mean_filter_correction | 0.0637 | [0.0606, 0.0662] |
| solver_success_fraction | 1.0000 | [1.0000, 1.0000] |
| constraint_max_violation | 0.0000 | [0.0000, 0.0000] |
| mean_solver_time_ms | 0.4335 | [0.4247, 0.4426] |
| safety_distance | 0.2347 | [0.2347, 0.2347] |
| speed_limit_mps | 0.2500 | [0.2500, 0.2500] |

## Paired treatment differences

Differences are treatment minus baseline.

| Baseline | Metric | Mean difference | 95% bootstrap CI |
|---|---|---:|---:|
| cyclic_projection | minimum_pairwise_distance | -0.0010 | [-0.0022, 0.0002] |
| cyclic_projection | collision_step_fraction | 0.0000 | [0.0000, 0.0000] |
| cyclic_projection | safety_violation_step_fraction | 0.0192 | [-0.0091, 0.0424] |
| cyclic_projection | goal_success_fraction | -0.0083 | [-0.0250, 0.0000] |
| cyclic_projection | final_goal_rmse | 0.0005 | [-0.0036, 0.0047] |
| cyclic_projection | normalized_command_energy | 0.0003 | [-0.0009, 0.0014] |
| cyclic_projection | mean_filter_correction | 0.0003 | [-0.0013, 0.0020] |
| cyclic_projection | solver_success_fraction | 0.0000 | [0.0000, 0.0000] |
| cyclic_projection | constraint_max_violation | 0.0000 | [0.0000, 0.0000] |
| cyclic_projection | mean_solver_time_ms | -0.3377 | [-0.3910, -0.2922] |
| cyclic_projection | safety_distance | 0.0000 | [0.0000, 0.0000] |
| cyclic_projection | speed_limit_mps | 0.0000 | [0.0000, 0.0000] |
| no_filter | minimum_pairwise_distance | 0.0905 | [0.0824, 0.0983] |
| no_filter | collision_step_fraction | -0.0076 | [-0.0196, -0.0004] |
| no_filter | safety_violation_step_fraction | -0.2952 | [-0.3241, -0.2574] |
| no_filter | goal_success_fraction | 0.0667 | [0.0250, 0.1167] |
| no_filter | final_goal_rmse | -0.0243 | [-0.0386, -0.0153] |
| no_filter | normalized_command_energy | -0.0137 | [-0.0153, -0.0121] |
| no_filter | mean_filter_correction | 0.0637 | [0.0606, 0.0662] |
| no_filter | solver_success_fraction | 0.5423 | [0.5251, 0.5614] |
| no_filter | constraint_max_violation | -0.1393 | [-0.1500, -0.1307] |
| no_filter | mean_solver_time_ms | 0.4335 | [0.4247, 0.4426] |
| no_filter | safety_distance | 0.0000 | [0.0000, 0.0000] |
| no_filter | speed_limit_mps | 0.0000 | [0.0000, 0.0000] |

## Interpretation guardrails

- Smoke and five-seed pilot intervals are diagnostic, not confirmatory.
- A rigid-body simulation improves dynamics validity but does not establish sim-to-real safety.
- The evaluated controller uses structured objective profiles; independent-language evidence is separate.
- Collision distance and linearized command-space constraints are reported separately.

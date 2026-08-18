# Betaflight SITL Result Card: betaflight_sitl_closed_loop_smoke_v19

- Status: `complete`
- Topology: Betaflight in WSL2; PyBullet in isolated Windows Conda
- Scope: single-drone flight-controller SITL; not multi-UAV SITL, HIL, or real flight

## Acceptance

- PASS: `motor_packet_receive_fraction`
- PASS: `finite_trajectory`
- PASS: `motor_output_nonzero`
- PASS: `altitude_response`

## Metrics

- `state_packets_sent`: 3000.0
- `rc_packets_sent`: 3000.0
- `motor_packets_received`: 2971.0
- `invalid_motor_packets`: 0.0
- `motor_packet_receive_fraction`: 0.9903333333333333
- `elapsed_seconds`: 34.15134099998977
- `initial_altitude`: 0.0999608
- `final_altitude`: 0.0630050813841443
- `maximum_altitude`: 0.24865056945241168
- `minimum_altitude`: 0.013335637016322809
- `final_position_error`: 9.092642931509982
- `motor_output_max`: 1.0
- `motor_output_mean_after_arm`: 0.37402480098716917
- `finite_trajectory`: True
- `sitl_process_exit_code`: 0

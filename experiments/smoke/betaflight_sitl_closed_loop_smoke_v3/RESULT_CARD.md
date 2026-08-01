# Betaflight SITL Result Card: betaflight_sitl_closed_loop_smoke_v3

- Status: `failed`
- Topology: Betaflight in WSL2; PyBullet in isolated Windows Conda
- Scope: single-drone flight-controller SITL; not multi-UAV SITL, HIL, or real flight

## Acceptance

- PASS: `motor_packet_receive_fraction`
- PASS: `finite_trajectory`
- FAIL: `motor_output_nonzero`
- FAIL: `altitude_response`

## Metrics

- `state_packets_sent`: 3000.0
- `rc_packets_sent`: 3000.0
- `motor_packets_received`: 2972.0
- `invalid_motor_packets`: 0.0
- `motor_packet_receive_fraction`: 0.9906666666666667
- `elapsed_seconds`: 14.128454599995166
- `initial_altitude`: 0.0999608
- `final_altitude`: 0.013490102373849856
- `maximum_altitude`: 0.0999608
- `minimum_altitude`: 0.013191472436448031
- `final_position_error`: 0.9865098976276657
- `motor_output_max`: 0.0
- `motor_output_mean_after_arm`: 0.0
- `finite_trajectory`: True
- `sitl_process_exit_code`: 124

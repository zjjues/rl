"""Run a checksummed multi-drone Betaflight-PyBullet SITL closed-loop smoke."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from betaflight_sitl_bridge import (
    BetaflightUdpBridge,
    motor_thrust_to_rpm,
    resolve_wsl_network,
    ports_for_drone,
)


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def checksums(root: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}")
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] != "--config":
        raise SystemExit("usage: run_betaflight_sitl_multi_smoke.py --config CONFIG.json")
    config_path = Path(sys.argv[2])
    config_path = config_path.resolve() if config_path.is_absolute() else (ROOT / config_path).resolve()
    spec = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / "experiments" / str(spec["level"]) / str(spec["study_id"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing study: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", spec)

    wsl = str(spec["wsl_executable"])
    distribution = str(spec["wsl_distribution"])
    sitl_root = str(spec["betaflight_root"])
    num_drones = int(spec["num_drones"])

    guest_ip, host_ip = resolve_wsl_network(wsl, distribution)
    duration = float(spec["duration_seconds"])
    frequency = int(spec["frequency_hz"])
    arm_seconds = float(spec["arm_seconds"])
    traj_start = float(spec["trajectory_start_seconds"])
    target_pos = np.asarray(spec["target_position"], dtype=np.float64)
    process_timeout = int(np.ceil(duration * 4.0 + 10.0))

    from gym_pybullet_drones.control.CTBRControl import CTBRControl
    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
    from gym_pybullet_drones.utils.enums import DroneModel, Physics

    # --- Start SITL instances ---
    sitl_processes = []
    sitl_logs = []
    for drone_id in range(num_drones):
        port_offset = drone_id * 10
        sitl_dir = f"/home/zhaji/rl-sitl/bf{drone_id}"
        cmd = (
            f"cd {shlex.quote(sitl_dir)} && "
            f"script -q -c 'timeout {process_timeout}s ./betaflight_SITL.elf "
            f"--ip {shlex.quote(host_ip)} --port-offset {port_offset}' /dev/null"
        )
        log_file = (output / f"sitl_stdout_drone{drone_id}.log").open("w", encoding="utf-8")
        sitl_logs.append(log_file)
        proc = subprocess.Popen(
            [wsl, "-d", distribution, "--", "bash", "-lc", cmd],
            stdout=log_file, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        sitl_processes.append(proc)

    # --- Wait for SITL init ---
    time.sleep(3.0)
    for i, proc in enumerate(sitl_processes):
        if proc.poll() is not None:
            raise RuntimeError(f"SITL drone {i} exited before the bridge started")

    # --- Create bridges ---
    bridges = []
    for drone_id in range(num_drones):
        bridge = BetaflightUdpBridge(guest_ip, timeout=2.0 / frequency, drone_id=drone_id)
        bridges.append(bridge)

    # --- Create PyBullet environment ---
    initial_xyzs = np.asarray([
        [float(i) * 0.5, 0.0, 0.10] for i in range(num_drones)
    ])
    env = CtrlAviary(
        drone_model=DroneModel.RACE, num_drones=num_drones,
        initial_xyzs=initial_xyzs,
        physics=Physics.PYB, pyb_freq=frequency, ctrl_freq=frequency,
        gui=False, record=False, user_debug_gui=False,
    )
    controller = CTBRControl(drone_model=DroneModel.RACE)
    observation, _ = env.reset(seed=int(spec["seed"]))

    # --- Run simulation ---
    motor_thrusts = [np.zeros(4, dtype=np.float64) for _ in range(num_drones)]
    command_ctbr = [np.zeros(4, dtype=np.float64) for _ in range(num_drones)]
    positions = [[] for _ in range(num_drones)]
    motor_values = [[] for _ in range(num_drones)]
    started = time.perf_counter()

    try:
        for index in range(int(duration * frequency)):
            timestamp = index / frequency
            armed = timestamp >= arm_seconds

            # Convert and step PyBullet
            rpms = np.zeros((num_drones, 4), dtype=np.float64)
            for j in range(num_drones):
                rpms[j] = motor_thrust_to_rpm(motor_thrusts[j], env.MAX_THRUST, env.KF)
            observation, _, _, _, _ = env.step(rpms)

            # Compute CTBR commands and exchange with bridges
            for j in range(num_drones):
                if timestamp > traj_start:
                    drone_target = target_pos.copy()
                    drone_target[0] += float(j) * 0.5
                    command_ctbr[j] = controller.computeControlFromState(
                        control_timestep=env.CTRL_TIMESTEP,
                        state=observation[j],
                        target_pos=drone_target,
                    )
                motor_thrusts[j], _ = bridges[j].exchange(
                    timestamp, observation[j], command_ctbr[j],
                    armed=armed, previous_motor_thrust=motor_thrusts[j],
                )
                positions[j].append(observation[j, :3].astype(float).tolist())
                motor_values[j].append(motor_thrusts[j].astype(float).tolist())

    finally:
        for bridge in bridges:
            bridge.close()
        env.close()
        for log in sitl_logs:
            log.close()
        for proc in sitl_processes:
            try:
                proc.wait(timeout=process_timeout + 5)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=5)

    # --- Aggregate results ---
    all_criteria = []
    for j in range(num_drones):
        pos_arr = np.asarray(positions[j], dtype=np.float64)
        mot_arr = np.asarray(motor_values[j], dtype=np.float64)
        audit = bridges[j].audit.as_dict()
        metrics = {
            **audit,
            "drone_id": j,
            "elapsed_seconds": float(time.perf_counter() - started),
            "initial_altitude": float(pos_arr[0, 2]),
            "final_altitude": float(pos_arr[-1, 2]),
            "maximum_altitude": float(pos_arr[:, 2].max()),
            "minimum_altitude": float(pos_arr[:, 2].min()),
            "final_position_error": float(np.linalg.norm(pos_arr[-1] - target_pos)),
            "motor_output_max": float(mot_arr.max()),
            "motor_output_mean_after_arm": float(mot_arr[int(frequency * arm_seconds):].mean()),
            "finite_trajectory": bool(np.isfinite(pos_arr).all() and np.isfinite(mot_arr).all()),
        }
        criteria = {
            "motor_packet_receive_fraction": metrics["motor_packet_receive_fraction"] >= 0.80,
            "finite_trajectory": metrics["finite_trajectory"],
            "motor_output_nonzero": metrics["motor_output_max"] > 0.05,
            "altitude_response": metrics["maximum_altitude"] > metrics["initial_altitude"] + 0.10,
        }
        all_criteria.append(criteria)
        write_json(output / f"result_drone{j}.json", {
            "status": "complete" if all(criteria.values()) else "failed",
            "metrics": metrics, "acceptance_criteria": criteria,
        })

    overall = all(all(c.values()) for c in all_criteria)
    result = {"status": "complete" if overall else "failed", "num_drones": num_drones,
              "drone_results": [f"drone{j}: {'PASS' if all(c.values()) else 'FAIL'}" for j, c in enumerate(all_criteria)]}

    # Manifest
    manifest = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "betaflight_revision": spec["betaflight_revision"],
        "simulator_revision": spec["gym_pybullet_drones_revision"],
        "python": sys.version, "platform": platform.platform(),
        "num_drones": num_drones, "scope": "multi-drone Betaflight SITL with PyBullet physics",
    }
    write_json(output / "result.json", result)
    write_json(output / "manifest.json", manifest)
    checksums(output)

    print(json.dumps({"output": str(output), **result}, indent=2))
    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

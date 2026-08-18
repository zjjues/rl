"""Run a checksummed single-drone Betaflight-PyBullet SITL closed-loop smoke."""

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
        raise SystemExit("usage: run_betaflight_sitl_smoke.py --config CONFIG.json")
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
    if not sitl_root.startswith("/home/") or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/_-." for ch in sitl_root):
        raise ValueError("betaflight_root must be a simple absolute path under /home")
    guest_ip, host_ip = resolve_wsl_network(wsl, distribution)
    duration = float(spec["duration_seconds"])
    frequency = int(spec["frequency_hz"])
    process_timeout = int(np.ceil(duration * 4.0 + 10.0))  # generous margin for PyBullet real-time
    command = (
        f"cd {shlex.quote(sitl_root)} && "
        f"script -q -c 'timeout {process_timeout}s ./obj/main/betaflight_SITL.elf --ip {shlex.quote(host_ip)}' /dev/null"
    )

    from gym_pybullet_drones.control.CTBRControl import CTBRControl
    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
    from gym_pybullet_drones.utils.enums import DroneModel, Physics

    # Pre-load: provision EEPROM via --config (one-shot, exits after save).
    config_path = spec.get("betaflight_config")
    if config_path:
        config_abs = f"{sitl_root}/{config_path}"
        provision_cmd = (
            f"cd {shlex.quote(sitl_root)} && "
            f"timeout 15s ./obj/main/betaflight_SITL.elf --ip {shlex.quote(host_ip)} "
            f"--config {shlex.quote(config_path)}"
        )
        provision_result = subprocess.run(
            [wsl, "-d", distribution, "--", "bash", "-lc", provision_cmd],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            timeout=20,
        )
        # Log the provision output for diagnostics
        stdout_text = provision_result.stdout or ""
        stderr_text = provision_result.stderr or ""
        provision_log = (output / "sitl_provision.log")
        provision_log.write_text(stdout_text + "\n" + stderr_text, encoding="utf-8")
        if provision_result.returncode not in (0, 124):  # 124 = timeout (expected)
            raise RuntimeError(
                f"SITL --config provisioning failed (exit {provision_result.returncode}):\n"
                f"{stderr_text[:500]}"
            )

    log_file = (output / "sitl_stdout.log").open("w", encoding="utf-8")
    sitl_process = subprocess.Popen(
        [wsl, "-d", distribution, "--", "bash", "-lc", command],
        stdout=log_file, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    bridge = None
    env = None
    positions, motor_values = [], []
    started = time.perf_counter()
    try:
        time.sleep(3.0)  # allow SITL flash init to complete before sending packets
        if sitl_process.poll() is not None:
            raise RuntimeError("Betaflight SITL exited before the bridge started")
        bridge = BetaflightUdpBridge(guest_ip, timeout=2.0 / frequency)
        env = CtrlAviary(
            drone_model=DroneModel.RACE, num_drones=1,
            initial_xyzs=np.asarray([[0.0, 0.0, 0.10]]),
            physics=Physics.PYB, pyb_freq=frequency, ctrl_freq=frequency,
            gui=False, record=False, user_debug_gui=False,
        )
        controller = CTBRControl(drone_model=DroneModel.RACE)
        observation, _ = env.reset(seed=int(spec["seed"]))
        motor_thrust = np.zeros(4, dtype=np.float64)
        command_ctbr = np.zeros(4, dtype=np.float64)
        for index in range(int(duration * frequency)):
            timestamp = index / frequency
            rpm = motor_thrust_to_rpm(motor_thrust, env.MAX_THRUST, env.KF)
            observation, _, _, _, _ = env.step(rpm.reshape(1, 4))
            if timestamp > float(spec["trajectory_start_seconds"]):
                command_ctbr = controller.computeControlFromState(
                    control_timestep=env.CTRL_TIMESTEP,
                    state=observation[0],
                    target_pos=np.asarray(spec["target_position"], dtype=np.float64),
                )
            motor_thrust, _ = bridge.exchange(
                timestamp, observation[0], command_ctbr,
                armed=timestamp >= float(spec["arm_seconds"]),
                previous_motor_thrust=motor_thrust,
            )
            positions.append(observation[0, :3].astype(float).tolist())
            motor_values.append(motor_thrust.astype(float).tolist())
    finally:
        if bridge is not None:
            bridge.close()
        if env is not None:
            env.close()
        try:
            sitl_process.wait(timeout=process_timeout + 2)
        except subprocess.TimeoutExpired:
            sitl_process.terminate()
            sitl_process.wait(timeout=5)
        log_file.close()

    position_array = np.asarray(positions, dtype=np.float64)
    motor_array = np.asarray(motor_values, dtype=np.float64)
    audit = bridge.audit.as_dict() if bridge is not None else {}
    metrics = {
        **audit,
        "elapsed_seconds": float(time.perf_counter() - started),
        "initial_altitude": float(position_array[0, 2]),
        "final_altitude": float(position_array[-1, 2]),
        "maximum_altitude": float(position_array[:, 2].max()),
        "minimum_altitude": float(position_array[:, 2].min()),
        "final_position_error": float(np.linalg.norm(position_array[-1] - np.asarray(spec["target_position"]))),
        "motor_output_max": float(motor_array.max()),
        "motor_output_mean_after_arm": float(motor_array[int(frequency * spec["arm_seconds"]):].mean()),
        "finite_trajectory": bool(np.isfinite(position_array).all() and np.isfinite(motor_array).all()),
        "sitl_process_exit_code": int(sitl_process.returncode),
    }
    criteria = {
        "motor_packet_receive_fraction": metrics["motor_packet_receive_fraction"] >= 0.80,
        "finite_trajectory": metrics["finite_trajectory"],
        "motor_output_nonzero": metrics["motor_output_max"] > 0.05,
        "altitude_response": metrics["maximum_altitude"] > metrics["initial_altitude"] + 0.10,
    }
    result = {"status": "complete" if all(criteria.values()) else "failed", "metrics": metrics,
              "acceptance_criteria": criteria}
    write_json(output / "result.json", result)
    binary_hash = subprocess.check_output(
        [wsl, "-d", distribution, "--", "sha256sum", f"{sitl_root}/obj/main/betaflight_SITL.elf"],
        text=True,
    ).split()[0]
    manifest = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "betaflight_revision": spec["betaflight_revision"], "betaflight_binary_sha256": binary_hash,
        "simulator_revision": spec["gym_pybullet_drones_revision"],
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in (
            "numpy", "pybullet", "gym-pybullet-drones", "control", "transforms3d"
        )},
        "network": {"wsl_guest_ip": guest_ip, "windows_host_ip_from_wsl": host_ip,
                    "pwm_port": 9002, "state_port": 9003, "rc_port": 9004},
        "scope": "single-drone Betaflight flight-controller SITL with PyBullet physics; not HIL or real flight",
    }
    write_json(output / "manifest.json", manifest)
    card = [f"# Betaflight SITL Result Card: {spec['study_id']}", "",
            f"- Status: `{result['status']}`", "- Topology: Betaflight in WSL2; PyBullet in isolated Windows Conda",
            "- Scope: single-drone flight-controller SITL; not multi-UAV SITL, HIL, or real flight", "",
            "## Acceptance", ""]
    card.extend(f"- {'PASS' if passed else 'FAIL'}: `{name}`" for name, passed in criteria.items())
    card.extend(["", "## Metrics", ""])
    card.extend(f"- `{name}`: {value}" for name, value in metrics.items())
    (output / "RESULT_CARD.md").write_text("\n".join(card) + "\n", encoding="utf-8")
    checksums(output)
    print(json.dumps({"output": str(output), **result}, indent=2))
    if result["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

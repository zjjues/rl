"""Explicit UDP bridge between Windows PyBullet and Betaflight SITL in WSL."""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


PORT_PWM = 9002
PORT_STATE = 9003
PORT_RC = 9004


def quaternion_xyzw_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm <= 1e-12:
        return np.eye(3)
    x, y, z, w = np.asarray([x, y, z, w]) / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def pack_fdm_state(timestamp: float, observation: np.ndarray) -> bytes:
    """Pack the upstream 18-double Betaflight FDM message."""
    observation = np.asarray(observation, dtype=np.float64)
    rotation = quaternion_xyzw_to_rotation(observation[3:7])
    angular_body = rotation.T @ observation[13:16]
    return struct.pack(
        "@18d", float(timestamp),
        float(angular_body[0]), float(-angular_body[1]), float(-angular_body[2]),
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    )


def ctbr_to_rc(
    timestamp: float,
    command: np.ndarray,
    *,
    armed: bool,
    max_thrust: float = 40.9,
    max_rate_degrees: float = 360.0,
) -> bytes:
    """Convert collective-thrust/body-rate command to Betaflight RC packet."""
    thrust, roll, pitch, yaw = np.asarray(command, dtype=np.float64)
    thrust_channel = np.clip(thrust / max_thrust * 1000.0 + 1000.0, 1000.0, 2000.0)
    rates = np.asarray([roll, pitch, -yaw]) * 180.0 / np.pi
    rates = np.clip(rates / max_rate_degrees * 500.0 + 1500.0, 1000.0, 2000.0)
    channels = [
        int(round(rates[0])), int(round(rates[1])), int(round(thrust_channel)),
        int(round(rates[2])), 1500 if armed else 1000,
    ] + [1000] * 11
    return struct.pack("@d16H", float(timestamp), *channels)


def decode_motor_packet(packet: bytes) -> np.ndarray:
    if len(packet) != 16:
        raise ValueError(f"expected a 16-byte motor packet, received {len(packet)}")
    values = np.asarray(struct.unpack("@4f", packet), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("motor packet contains non-finite values")
    return values


def motor_thrust_to_rpm(motor_thrust: np.ndarray, max_thrust: float, kf: float) -> np.ndarray:
    """Apply the upstream Betaflight-to-PyBullet motor order and scaling."""
    motor_thrust = np.asarray(motor_thrust, dtype=np.float64)
    if motor_thrust.shape != (4,) or max_thrust <= 0.0 or kf <= 0.0:
        raise ValueError("invalid motor thrust conversion inputs")
    remapped = motor_thrust[[2, 1, 3, 0]]
    return np.sqrt(max_thrust / 4.0 / kf * np.clip(remapped, 0.0, None))


@dataclass
class BridgeAudit:
    state_packets_sent: int = 0
    rc_packets_sent: int = 0
    motor_packets_received: int = 0
    invalid_motor_packets: int = 0

    def as_dict(self) -> Dict[str, float]:
        denominator = max(self.state_packets_sent, 1)
        return {
            "state_packets_sent": float(self.state_packets_sent),
            "rc_packets_sent": float(self.rc_packets_sent),
            "motor_packets_received": float(self.motor_packets_received),
            "invalid_motor_packets": float(self.invalid_motor_packets),
            "motor_packet_receive_fraction": float(self.motor_packets_received / denominator),
        }


class BetaflightUdpBridge:
    """Own UDP sockets for one externally managed Betaflight SITL process."""

    def __init__(self, sitl_ip: str, bind_ip: str = "0.0.0.0", timeout: float = 0.003):
        self.sitl_ip = str(sitl_ip)
        self.state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rc_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.motor_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.motor_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.motor_socket.bind((bind_ip, PORT_PWM))
        self.motor_socket.settimeout(float(timeout))
        self.audit = BridgeAudit()

    def exchange(
        self,
        timestamp: float,
        observation: np.ndarray,
        command: np.ndarray,
        armed: bool,
        previous_motor_thrust: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        self.state_socket.sendto(pack_fdm_state(timestamp, observation), (self.sitl_ip, PORT_STATE))
        self.audit.state_packets_sent += 1
        self.rc_socket.sendto(ctbr_to_rc(timestamp, command, armed=armed), (self.sitl_ip, PORT_RC))
        self.audit.rc_packets_sent += 1
        try:
            packet, _ = self.motor_socket.recvfrom(64)
        except socket.timeout:
            return np.asarray(previous_motor_thrust, dtype=np.float64), False
        try:
            motor_thrust = decode_motor_packet(packet)
        except ValueError:
            self.audit.invalid_motor_packets += 1
            return np.asarray(previous_motor_thrust, dtype=np.float64), False
        self.audit.motor_packets_received += 1
        return motor_thrust, True

    def close(self) -> None:
        self.state_socket.close()
        self.rc_socket.close()
        self.motor_socket.close()


def resolve_wsl_network(wsl_executable: str, distribution: str) -> Tuple[str, str]:
    """Return (WSL guest IP, Windows host IP as seen from WSL)."""
    import subprocess

    guest = subprocess.check_output(
        [wsl_executable, "-d", distribution, "--", "hostname", "-I"], text=True
    ).strip().split()[0]
    route = subprocess.check_output(
        [wsl_executable, "-d", distribution, "--", "ip", "route", "show", "default"],
        text=True,
    ).strip().split()
    host = route[route.index("via") + 1]
    return guest, host

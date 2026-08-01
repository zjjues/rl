import struct
import unittest

import numpy as np

from src.betaflight_sitl_bridge import (
    ctbr_to_rc,
    decode_motor_packet,
    motor_thrust_to_rpm,
    pack_fdm_state,
)


class BetaflightSITLBridgeTest(unittest.TestCase):
    def test_packet_sizes_match_upstream_protocol(self):
        observation = np.zeros(20)
        observation[6] = 1.0
        self.assertEqual(len(pack_fdm_state(0.0, observation)), 18 * 8)
        self.assertEqual(len(ctbr_to_rc(0.0, np.zeros(4), armed=False)), 8 + 16 * 2)

    def test_rc_arm_and_neutral_channels(self):
        values = struct.unpack("@d16H", ctbr_to_rc(1.0, np.zeros(4), armed=True))
        self.assertEqual(values[1:5], (1500, 1500, 1000, 1500))
        self.assertEqual(values[5], 1500)

    def test_motor_packet_decode_and_order(self):
        decoded = decode_motor_packet(struct.pack("@4f", 1.0, 2.0, 3.0, 4.0))
        np.testing.assert_allclose(decoded, [1, 2, 3, 4])
        rpm = motor_thrust_to_rpm(decoded, max_thrust=4.0, kf=1.0)
        np.testing.assert_allclose(rpm, np.sqrt([3, 2, 4, 1]))

    def test_invalid_motor_packet_is_rejected(self):
        with self.assertRaises(ValueError):
            decode_motor_packet(b"short")


if __name__ == "__main__":
    unittest.main()

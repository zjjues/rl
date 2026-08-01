import unittest

import numpy as np

from src.pybullet_transfer import (
    crossing_scenario,
    latency_robust_margin,
    project_velocity_commands,
    safety_distance,
    velocity_constraint_diagnostics,
    velocity_to_aviary_action,
)


class PyBulletTransferTest(unittest.TestCase):
    def test_safety_contract_cannot_be_relaxed_by_profile(self):
        base = 0.22
        self.assertEqual(safety_distance({"safety": 0.3, "collision": 0.3}, base), base)
        self.assertGreater(safety_distance({"safety": 2.0}, base), base)

    def test_latency_margin_bounds_two_vehicle_closure(self):
        self.assertAlmostEqual(latency_robust_margin(0.25, 0.08), 0.04)

    def test_velocity_action_preserves_direction_and_fraction(self):
        commands = np.asarray([[0.1, 0.0, 0.0], [0.0, -0.2, 0.0]])
        action = velocity_to_aviary_action(commands, 0.25)
        np.testing.assert_allclose(action[:, :3], [[1, 0, 0], [0, -1, 0]])
        np.testing.assert_allclose(action[:, 3], [0.4, 0.8])

    def test_qp_projection_satisfies_feasible_pairwise_row(self):
        positions = np.asarray([[-0.12, 0.0, 1.0], [0.12, 0.0, 1.0]])
        velocities = np.zeros((2, 3))
        nominal = np.asarray([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]])
        filtered, audit = project_velocity_commands(
            positions, velocities, nominal, min_distance=0.22,
            max_speed=0.25, horizon=0.35, mode="qp",
        )
        measured = velocity_constraint_diagnostics(
            positions, velocities, filtered, min_distance=0.22, horizon=0.35,
        )
        self.assertEqual(audit["solver_success"], 1.0)
        self.assertLessEqual(measured["constraint_max_violation"], 1e-6)

    def test_scenario_is_seed_deterministic(self):
        first = crossing_scenario(7)
        second = crossing_scenario(7)
        np.testing.assert_allclose(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])


if __name__ == "__main__":
    unittest.main()

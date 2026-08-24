from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imappo import IMAPPO, IMAPPOConfig  # noqa: E402
from rule_based_baseline import (  # noqa: E402
    RuleBasedUAVPolicy,
    apply_pairwise_cbf_filter,
    apply_pairwise_cbf_filter_with_diagnostics,
    apply_pairwise_qp_filter,
    compute_rule_actions,
    pairwise_cbf_constraint_diagnostics,
)
from intent_objectives import resolve_intent_reward_profile  # noqa: E402


class RuleBasedBaselineTests(unittest.TestCase):
    @staticmethod
    def _legacy_cbf_reference(
        observations,
        actions,
        objective_profile=None,
        *,
        dt=0.2,
        velocity_retention=0.7,
        action_gain=0.3,
        base_min_distance=1.0,
        iterations=4,
    ):
        profile = objective_profile or {}
        safety = max(min(max(profile.get("safety", 1.0), 1.0), 2.2), 1.0)
        collision = max(min(max(profile.get("collision", 1.0), 1.0), 2.2), 1.0)
        min_distance = max(
            0.70 * base_min_distance,
            min(
                base_min_distance
                * (1.0 + 0.25 * (safety - 1.0) + 0.25 * (collision - 1.0)),
                1.60 * base_min_distance,
            ),
        )
        positions = observations[:, 0:3]
        velocities = observations[:, 3:6]
        filtered = actions.clone()
        for _ in range(iterations):
            for i in range(len(filtered)):
                for j in range(i + 1, len(filtered)):
                    relative_position = positions[i] - positions[j]
                    distance = torch.linalg.vector_norm(relative_position)
                    if float(distance.item()) <= 1e-6:
                        direction = torch.zeros_like(relative_position)
                        direction[0] = 1.0
                    else:
                        direction = relative_position / distance
                    required = (
                        min_distance
                        - float(distance.item())
                        - dt
                        * velocity_retention
                        * torch.dot(direction, velocities[i] - velocities[j])
                    ) / (dt * action_gain)
                    violation = required - torch.dot(
                        direction, filtered[i] - filtered[j]
                    )
                    if float(violation.item()) > 0.0:
                        correction = 0.5 * violation * direction
                        filtered[i] = filtered[i] + correction
                        filtered[j] = filtered[j] - correction
                        filtered = torch.clamp(filtered, -1.0, 1.0)
        return filtered

    def test_vectorized_cbf_matches_legacy_cyclic_projection(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            generator = torch.Generator(device="cpu").manual_seed(20260820)
            for n_agents in (1, 2, 8):
                observations = torch.randn(n_agents, 18, generator=generator).to(device)
                actions = torch.empty(n_agents, 3).uniform_(
                    -1.0, 1.0, generator=generator
                ).to(device)
                if n_agents >= 2:
                    observations[1, 0:3] = observations[0, 0:3]
                for profile in (None, {"safety": 1.7, "collision": 1.4}):
                    expected = self._legacy_cbf_reference(
                        observations, actions, profile, iterations=4
                    )
                    actual = apply_pairwise_cbf_filter(
                        observations, actions, profile, iterations=4
                    )
                    self.assertTrue(
                        torch.allclose(actual, expected, atol=2e-6, rtol=1e-6),
                        msg=(device, n_agents, profile, (actual - expected).abs().max()),
                    )

    def test_fused_cbf_diagnostics_match_independent_audit(self):
        observations = torch.randn(8, 18, generator=torch.Generator().manual_seed(17))
        actions = torch.empty(8, 3).uniform_(
            -1.0, 1.0, generator=torch.Generator().manual_seed(29)
        )
        filtered, fused = apply_pairwise_cbf_filter_with_diagnostics(
            observations, actions, {"safety": 1.5, "collision": 1.2}
        )
        independent = pairwise_cbf_constraint_diagnostics(
            observations, filtered, {"safety": 1.5, "collision": 1.2}
        )
        for key in independent:
            self.assertAlmostEqual(fused[key], independent[key], places=5)

    def test_qp_cbf_solves_feasible_box_constrained_projection(self):
        observations = torch.zeros(2, 18)
        observations[1, 0] = 0.9
        closing_actions = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        filtered, solver = apply_pairwise_qp_filter(
            observations, closing_actions, tolerance=1e-6
        )
        audit = pairwise_cbf_constraint_diagnostics(observations, filtered)
        self.assertEqual(solver["safety_filter_solver_success"], 1.0)
        self.assertLessEqual(audit["cbf_constraint_max_violation"], 1e-6)
        self.assertTrue((filtered.abs() <= 1.0 + 1e-7).all().item())

    def test_qp_cbf_reports_infeasible_action_box(self):
        observations = torch.zeros(2, 18)
        observations[1, 0] = 0.01
        filtered, solver = apply_pairwise_qp_filter(
            observations,
            torch.zeros(2, 3),
            {"safety": 2.2},
            tolerance=1e-6,
        )
        audit = pairwise_cbf_constraint_diagnostics(
            observations, filtered, {"safety": 2.2}
        )
        self.assertEqual(solver["safety_filter_solver_success"], 0.0)
        self.assertGreater(audit["cbf_constraint_max_violation"], 0.0)

    def test_cbf_diagnostics_report_and_clear_linear_constraint_violation(self):
        observations = torch.zeros(2, 18)
        observations[1, 0] = 0.9
        closing_actions = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        before = pairwise_cbf_constraint_diagnostics(observations, closing_actions)
        filtered = apply_pairwise_cbf_filter(observations, closing_actions)
        after = pairwise_cbf_constraint_diagnostics(observations, filtered)
        self.assertGreater(before["cbf_constraint_max_violation"], 0.0)
        self.assertLessEqual(after["cbf_constraint_max_violation"], 1e-5)
        self.assertEqual(after["cbf_constraint_violation_fraction"], 0.0)

    def test_pairwise_cbf_projects_closing_actions_to_separation_constraint(self):
        observations = torch.zeros(2, 18)
        observations[1, 0] = 0.9
        closing_actions = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        filtered = apply_pairwise_cbf_filter(
            observations, closing_actions, base_min_distance=1.0
        )
        direction = torch.tensor([-1.0, 0.0, 0.0])
        radial_action = torch.dot(direction, filtered[0] - filtered[1])
        required = (1.0 - 0.9) / (0.2 * 0.3)
        self.assertGreaterEqual(float(radial_action), required - 1e-5)
        self.assertTrue((filtered.abs() <= 1.0).all().item())

    def test_high_safety_profile_triggers_stronger_cbf_correction(self):
        observations = torch.zeros(2, 18)
        observations[1, 0] = 1.05
        actions = torch.zeros(2, 3)
        low = apply_pairwise_cbf_filter(
            observations, actions, {"safety": 0.5, "collision": 0.5}
        )
        high = apply_pairwise_cbf_filter(
            observations, actions, {"safety": 1.7, "collision": 1.7}
        )
        self.assertGreater(
            float(torch.linalg.vector_norm(high - actions)),
            float(torch.linalg.vector_norm(low - actions)),
        )

    def test_language_cannot_relax_base_collision_barrier(self):
        observations = torch.zeros(2, 18)
        observations[1, 0] = 0.9
        actions = torch.zeros(2, 3)
        neutral = apply_pairwise_cbf_filter(observations, actions)
        requested_relaxation = apply_pairwise_cbf_filter(
            observations, actions, {"collision": 0.3, "safety": 0.3}
        )
        self.assertTrue(torch.allclose(neutral, requested_relaxation, atol=1e-6))

    def test_controller_moves_toward_target_without_neighbors(self):
        cfg = IMAPPOConfig(n_agents=2, obs_dim=18, state_dim=36, intent_dim=8)
        policy = RuleBasedUAVPolicy(cfg)
        observations = torch.zeros(2, 18)
        observations[:, 6] = 2.0
        mask = torch.ones(2, 3)
        actions, _ = policy.select_actions(
            observations, torch.zeros(8), mask, deterministic=True
        )
        self.assertTrue((actions[:, 0] > 0).all().item())

    def test_close_neighbor_creates_repulsion(self):
        cfg = IMAPPOConfig(n_agents=2, obs_dim=18, state_dim=36, intent_dim=8)
        policy = RuleBasedUAVPolicy(cfg)
        policy.set_evaluation_context("safety_first", "stealth")
        observations = torch.zeros(2, 18)
        observations[:, 6] = 2.0
        observations[:, 12] = 0.2
        mask = torch.ones(2, 3)
        actions, _ = policy.select_actions(
            observations, torch.zeros(8), mask, deterministic=True
        )
        self.assertTrue((actions[:, 0] < 0).all().item())

    def test_zero_initialized_residual_policy_matches_rule_prior(self):
        cfg = IMAPPOConfig(
            n_agents=2,
            n_targets=2,
            obs_dim=18,
            state_dim=36,
            intent_dim=25,
            intent_source="onehot",
            policy_mode="residual_rule",
            use_action_mask=False,
        )
        algo = IMAPPO(cfg)
        algo.set_evaluation_context("balanced", "neutral")
        observations = torch.zeros(2, 18)
        observations[:, 6] = 2.0
        mask = torch.ones(2, 3)
        actions, log_probs = algo.select_actions(
            observations, torch.zeros(25), mask, deterministic=True
        )
        expected = compute_rule_actions(observations, "neutral", mask)
        self.assertTrue(torch.allclose(actions, expected, atol=1.1e-3))
        self.assertTrue(torch.isfinite(log_probs).all().item())

    def test_residual_policy_can_deviate_from_prior(self):
        cfg = IMAPPOConfig(
            n_agents=2,
            n_targets=2,
            obs_dim=18,
            state_dim=36,
            intent_dim=25,
            intent_source="onehot",
            policy_mode="residual_rule",
            residual_action_scale=0.5,
            use_action_mask=False,
        )
        algo = IMAPPO(cfg)
        observations = torch.zeros(2, 18)
        observations[:, 6] = 2.0
        mask = torch.ones(2, 3)
        before, _ = algo.select_actions(
            observations, torch.zeros(25), mask, deterministic=True
        )
        with torch.no_grad():
            algo.actor.mean_head.bias[1] = 1.0
        after, _ = algo.select_actions(
            observations, torch.zeros(25), mask, deterministic=True
        )
        self.assertTrue((after[:, 1] > before[:, 1]).all().item())

    def test_residual_keeps_inward_gradient_at_saturated_rule_action(self):
        cfg = IMAPPOConfig(
            n_agents=2,
            n_targets=2,
            obs_dim=18,
            state_dim=36,
            intent_dim=25,
            intent_source="onehot",
            policy_mode="residual_rule",
            residual_action_scale=0.5,
            use_action_mask=False,
        )
        algo = IMAPPO(cfg)
        observations = torch.zeros(2, 18)
        observations[:, 6] = 2.0
        mask = torch.ones(2, 3)
        prior, _ = algo.select_actions(
            observations, torch.zeros(25), mask, deterministic=True
        )
        self.assertTrue(torch.allclose(prior[:, 0], torch.ones(2)))
        with torch.no_grad():
            algo.actor.mean_head.bias[0] = -1.0
        moved, _ = algo.select_actions(
            observations, torch.zeros(25), mask, deterministic=True
        )
        self.assertTrue((moved[:, 0] < prior[:, 0] - 0.5).all().item())

    def test_intent_retrieval_prior_does_not_use_external_posture_oracle(self):
        cfg = IMAPPOConfig(
            n_agents=2,
            n_targets=2,
            obs_dim=18,
            state_dim=36,
            intent_dim=25,
            intent_source="onehot",
            policy_mode="residual_rule",
            rule_prior_context="intent_retrieval",
            use_action_mask=False,
        )
        algo = IMAPPO(cfg)
        algo.set_evaluation_context("aggressive_pursuit", "attack")
        safety = torch.from_numpy(algo.intent_library.get_by_label("safety_first"))
        self.assertEqual(algo._rule_prior_posture(safety), "stealth")

    def test_neutral_prior_ignores_external_posture_oracle(self):
        cfg = IMAPPOConfig(
            n_agents=2,
            n_targets=2,
            obs_dim=18,
            state_dim=36,
            intent_dim=25,
            intent_source="onehot",
            policy_mode="residual_rule",
            rule_prior_context="neutral",
            use_action_mask=False,
        )
        algo = IMAPPO(cfg)
        algo.set_evaluation_context("safety_first", "stealth")
        self.assertEqual(algo._rule_prior_posture(torch.zeros(25)), "neutral")

    def test_decentralized_actor_executes_on_larger_swarm(self):
        cfg = IMAPPOConfig(
            n_agents=4,
            n_targets=4,
            obs_dim=30,
            state_dim=120,
            intent_dim=25,
            intent_source="onehot",
            policy_mode="residual_rule",
            use_action_mask=False,
        )
        algo = IMAPPO(cfg)
        observations = torch.zeros(12, 30)
        observations[:, 6] = 2.0
        actions, log_probs = algo.select_actions(
            observations, torch.zeros(25), torch.ones(12, 3), deterministic=True
        )
        self.assertEqual(tuple(actions.shape), (12, 3))
        self.assertEqual(tuple(log_probs.shape), (12,))

    def test_continuous_objective_profile_modulates_action_ceiling(self):
        observations = torch.zeros(2, 18)
        observations[:, 6] = 2.0
        mask = torch.ones(2, 3)
        energy_saving = compute_rule_actions(
            observations,
            action_mask=mask,
            objective_profile=resolve_intent_reward_profile("energy_saving"),
        )
        rapid_response = compute_rule_actions(
            observations,
            action_mask=mask,
            objective_profile=resolve_intent_reward_profile("rapid_response"),
        )
        self.assertTrue(
            (torch.linalg.vector_norm(energy_saving, dim=-1)
             < torch.linalg.vector_norm(rapid_response, dim=-1)).all().item()
        )

    def test_continuous_safety_profile_strengthens_close_neighbor_repulsion(self):
        observations = torch.zeros(2, 18)
        observations[:, 6] = 2.0
        observations[:, 12] = 0.8
        safety = compute_rule_actions(
            observations,
            objective_profile=resolve_intent_reward_profile("safety_first"),
        )
        aggressive = compute_rule_actions(
            observations,
            objective_profile=resolve_intent_reward_profile("aggressive_pursuit"),
        )
        self.assertTrue((safety[:, 0] < aggressive[:, 0]).all().item())

    def test_safety_profile_changes_proactive_repulsion_radius(self):
        observations = torch.zeros(2, 18)
        observations[:, 6] = 2.0
        observations[:, 12] = 2.6
        high = compute_rule_actions(
            observations, objective_profile={"safety": 1.7}
        )
        low = compute_rule_actions(
            observations, objective_profile={"safety": 0.5}
        )
        self.assertTrue((high[:, 0] < low[:, 0]).all().item())

    def test_threat_preference_reduces_motion_toward_threat_coincident_target(self):
        observations = torch.zeros(2, 18)
        observations[:, 6] = 2.0
        low_threat = compute_rule_actions(
            observations, objective_profile={"threat": 0.5}
        )
        high_threat = compute_rule_actions(
            observations, objective_profile={"threat": 1.7}
        )
        self.assertTrue((high_threat[:, 0] < low_threat[:, 0]).all().item())


if __name__ == "__main__":
    unittest.main()

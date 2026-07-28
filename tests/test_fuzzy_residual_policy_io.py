from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from envs import Arm2DOFDynamicEnvConfig
from rl import (
    FuzzyDynamicStateEncoder,
    FuzzyResidualQLearningConfig,
    FuzzyResidualSafetyConfig,
    load_fuzzy_residual_policy,
    save_fuzzy_residual_policy,
    train_fuzzy_residual_q_learning,
)


class FuzzyResidualPolicyIOTests(unittest.TestCase):
    def test_save_and_load_policy_preserves_table_and_deployment_config(self) -> None:
        encoder = FuzzyDynamicStateEncoder(
            error_scale=(0.9, 1.2),
            velocity_scale=(6.0, 6.0),
        )
        learning_config = FuzzyResidualQLearningConfig(
            episodes=2,
            max_steps_per_episode=3,
            residual_acceleration_scale=(1.5, 1.5),
            seed=3,
        )
        safety_config = FuzzyResidualSafetyConfig(
            patience=5,
            min_progress=1e-4,
        )
        result = train_fuzzy_residual_q_learning(
            Arm2DOFDynamicEnvConfig(max_steps=3),
            encoder=encoder,
            config=learning_config,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = Path(tmpdir) / "policy.npz"
            save_fuzzy_residual_policy(
                policy_path,
                result,
                learning_config,
                safety_config,
                "T1",
                (1.1, 0.55),
                (("T1", (1.1, 0.55)), ("T2", (0.85, 0.85))),
            )

            package = load_fuzzy_residual_policy(policy_path)

        np.testing.assert_allclose(package.q_value, result.q_value)
        np.testing.assert_array_equal(package.rule_policy, result.rule_policy)
        self.assertEqual(package.encoder.n_rules, encoder.n_rules)
        self.assertEqual(package.learning_config.episodes, 2)
        self.assertEqual(package.safety_config.patience, 5)
        self.assertEqual(package.metadata["train_target"]["id"], "T1")
        self.assertEqual(len(package.metadata["deployment_targets"]), 2)


if __name__ == "__main__":
    unittest.main()

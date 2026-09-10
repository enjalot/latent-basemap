"""Regression checks for the aggregate/arm-CI bugs, without model projection."""
import unittest
import numpy as np
from card005_gates import arrival_gate


class ArrivalGateTests(unittest.TestCase):
    def setUp(self):
        self.sources = np.repeat(["a", "b", "c"], 100)
        self.frozen = np.full(300, 0.5)

    def evaluate(self, delta):
        return arrival_gate(self.frozen + delta, self.frozen,
                            self.sources, ["a", "b", "c"])

    def test_aggregate_can_pass_when_one_source_is_below_threshold(self):
        self.assertTrue(self.evaluate(np.repeat([0.02, 0.04, 0.05], 100))["gate_new_source"])

    def test_positive_mean_requires_own_positive_interval(self):
        # Same passing mean, but enough paired variance that its CI crosses zero.
        self.assertTrue(self.evaluate(np.full(300, 0.04))["gate_new_source"])
        result = self.evaluate(np.tile([-0.46, 0.54], 150))
        self.assertGreater(result["arriving_agg_gain_B250"], 0.03)
        self.assertLess(result["arriving_agg_ci95_B250"][0], 0)
        self.assertFalse(result["gate_new_source"])

    def test_do_not_round_into_a_pass(self):
        self.assertFalse(self.evaluate(np.full(300, 0.029999))["gate_new_source"])

    def test_nonfinite_fails_closed(self):
        delta = np.full(300, 0.04)
        delta[0] = np.nan
        with self.assertRaises(ValueError):
            self.evaluate(delta)

    def test_unbalanced_seal_fails_closed(self):
        with self.assertRaises(ValueError):
            arrival_gate(self.frozen[:-1], self.frozen[:-1],
                         self.sources[:-1], ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()

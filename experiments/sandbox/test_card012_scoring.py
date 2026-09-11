"""Guard regressions: these exercise the functions used by the actual scorer."""
import unittest
import numpy as np
from score_card012 import quality_checks, deployment_checks, equal_cohort_means, all_finite


class ScoringContract(unittest.TestCase):
    def test_both_arrival_comparators_are_required(self):
        for uniform, anchored in [(False, True), (True, False), (False, False)]:
            q = quality_checks(True, uniform, anchored, .001)
            self.assertFalse(all(q.values()))
            self.assertFalse(deployment_checks(.001, .004, q)['DEPLOY_ELIGIBLE'])

    def test_every_quality_guard_applies_to_deployment(self):
        for old, gain in [(False, .001), (True, 0.), (True, -.001), (True, np.nan)]:
            q = quality_checks(old, True, True, gain)
            self.assertFalse(deployment_checks(.001, .004, q)['DEPLOY_ELIGIBLE'])

    def test_native_stability_limits_and_nonfinite(self):
        q = quality_checks(True, True, True, .001)
        self.assertTrue(deployment_checks(.01, .05, q)['DEPLOY_ELIGIBLE'])
        for mean, p99 in [(.010001, .04), (.001, .050001), (np.nan, .04), (.001, np.inf)]:
            self.assertFalse(deployment_checks(mean, p99, q)['DEPLOY_ELIGIBLE'])

    def test_equal_cohort_includes_ninth_source(self):
        by_source = {f's{i}': 0. for i in range(8)}
        by_source['diffusion-aesthetic-4k'] = 1.
        self.assertAlmostEqual(equal_cohort_means({'250': by_source})['250'], 1/9)

    def test_finite_check_reaches_beyond_first_100k(self):
        a = np.zeros((100002, 2), np.float32)
        self.assertTrue(all_finite(a))
        a[-1, 0] = np.nan
        self.assertFalse(all_finite(a))


if __name__ == '__main__':
    unittest.main()

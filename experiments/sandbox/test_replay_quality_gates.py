"""Regression checks for rounded and missing-data replay gate errors."""
import unittest
import numpy as np
from score_replay_reception import quality_gates

class QualityGatesTest(unittest.TestCase):
    def setUp(self):
        self.grp = np.array(['old'] * 100 + ['new'] * 100)
        self.pq = {h: {b: np.full(200, .5) for b in (250, 2000)} for h in ('in','out','frozen','anchored')}
        for h in ('in','out'):
            for b in (250,2000): self.pq[h][b][100:] += .04
    def gates(self, chinese=False):
        return quality_gates(self.pq,self.grp,['new'],['old'],chinese)
    def test_valid(self):
        a,b=self.gates(True)
        self.assertTrue(a['in']['pass'] and b['out']['pass'])
    def test_loss_is_not_rounded_into_pass(self):
        self.pq['out'][250][:100] -= .005001
        self.assertFalse(self.gates()[0]['out']['pass'])
    def test_chinese_gain_is_not_rounded_into_pass(self):
        self.pq['out'][250][100:] = .529999
        self.assertFalse(self.gates(True)[1]['out']['pass'])
    def test_missing_control_fails(self):
        del self.pq['anchored']
        with self.assertRaises(ValueError): self.gates()
    def test_missing_cohort_fails(self):
        with self.assertRaises(ValueError): quality_gates(self.pq,self.grp,[],['old'])
    def test_nonfinite_fails(self):
        self.pq['out'][250][0] = np.nan
        with self.assertRaises(ValueError): self.gates()

if __name__ == '__main__': unittest.main()

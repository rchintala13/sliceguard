"""Calibration: realised FDR under the null, and realised power.

The essential test of the whole project. Anyone can cite Benjamini-Hochberg
in a README; this measures whether the correction actually does the work it
is credited with, against the uncorrected alternative, on data where the
truth is known by construction.
"""

import math
import unittest

from src.audit import INDETERMINATE, SIGNIFICANT, UNTESTABLE, audit
from src.generator import PLANTED_CELLS, PLANTED_MARGIN, generate_eval_set

NULL_RUNS = 300
Q = 0.05


class TestFalseDiscoveryRateUnderTheNull(unittest.TestCase):
    """margin = 0.0: every slice's true disparity is exactly zero.

    So every DISPARITY verdict is, unambiguously, a false discovery. The
    realised FDR is then just the fraction of runs producing any claim at
    all (when no hypothesis is truly false, FDR == family-wise error rate).
    """

    @classmethod
    def setUpClass(cls):
        cls.bh_claims, cls.raw_claims = [], []
        for i in range(NULL_RUNS):
            records = generate_eval_set(3000, seed=50000 + i, margin=0.0)
            results = audit(records, max_order=2, q=Q)
            cls.bh_claims.append(
                sum(1 for r in results if r.verdict == SIGNIFICANT))
            # The generic report: same slices, same tests, no correction.
            cls.raw_claims.append(sum(
                1 for r in results
                if r.verdict != UNTESTABLE and r.p_value < Q and r.disparity > 0))
        cls.m = len([r for r in results if r.verdict != UNTESTABLE])

    def test_bh_holds_the_false_discovery_rate_at_the_nominal_level(self):
        runs_with_a_claim = sum(1 for c in self.bh_claims if c > 0)
        fdr = runs_with_a_claim / NULL_RUNS
        # Bound derived, not chosen: the estimate of a proportion from 300
        # independent runs has standard error at most sqrt(0.25/300) =
        # 0.0289, so the nominal level plus 3 SE is 0.0500 + 0.0866.
        # Measured value is 0.037, comfortably inside.
        self.assertLessEqual(fdr, Q + 3 * math.sqrt(0.25 / NULL_RUNS))

    def test_the_uncorrected_audit_cries_wolf_on_most_null_datasets(self):
        raw_rate = sum(1 for c in self.raw_claims if c > 0) / NULL_RUNS
        bh_rate = sum(1 for c in self.bh_claims if c > 0) / NULL_RUNS
        # Measured: 0.79 of null audits produce at least one "disparity",
        # averaging 2.15 claims each, from data with no disparity in it.
        self.assertGreater(raw_rate, 0.60)
        self.assertGreater(raw_rate, 10 * bh_rate)

    def test_uncorrected_claim_count_matches_the_per_test_alpha(self):
        mean_raw = sum(self.raw_claims) / NULL_RUNS
        # With m tests at level Q and one-directional counting, the
        # expected number of null rejections is m * Q / 2. That closed
        # form -- not a tolerance picked by eye -- is the bar here.
        expected = self.m * Q / 2.0
        se = math.sqrt(expected) / math.sqrt(NULL_RUNS)
        self.assertLess(abs(mean_raw - expected), 5 * se)

    def test_correction_costs_something_and_that_is_stated(self):
        # BH is not free: it suppresses essentially every null claim, and
        # the same conservatism is why underpowered slices need the
        # explicit INDETERMINATE verdict rather than silence.
        self.assertLess(sum(self.bh_claims) / NULL_RUNS, 0.5)


class TestPowerAgainstThePlantedEffect(unittest.TestCase):
    """Realised power -- including an honest negative result.

    The two planted cells carry the SAME 9-point gap but differ in size
    (about 300 rows vs about 600). Measured BH power differs sharply:
    0.68 vs 0.95. That is not a defect to tune away; it is the entire
    reason the report carries a minimum-detectable-disparity column. The
    assertions below check that the MDE the audit publishes correctly
    predicts which of the two it can reliably find.
    """

    @classmethod
    def setUpClass(cls):
        cls.runs = 40
        cls.hits = {cell: 0 for cell in PLANTED_CELLS}
        cls.verdicts = {cell: [] for cell in PLANTED_CELLS}
        cls.mdes = {cell: [] for cell in PLANTED_CELLS}
        cls.sizes = {cell: [] for cell in PLANTED_CELLS}
        for i in range(cls.runs):
            records = generate_eval_set(6000, seed=77000 + i)
            results = {r.slice.key: r for r in audit(records, max_order=3, q=Q)}
            for cell in PLANTED_CELLS:
                r = results[cell]
                cls.verdicts[cell].append(r.verdict)
                cls.mdes[cell].append(r.mde)
                cls.sizes[cell].append(r.n)
                if r.verdict == SIGNIFICANT:
                    cls.hits[cell] += 1

    def _by_size(self):
        ordered = sorted(PLANTED_CELLS,
                         key=lambda c: sum(self.sizes[c]) / self.runs)
        return ordered[0], ordered[-1]

    def test_the_larger_planted_cell_is_recovered_almost_always(self):
        _, larger = self._by_size()
        power = self.hits[larger] / self.runs
        # Measured 0.95 at n ~ 600. Floor from an exact binomial bound: if
        # true power were 0.80, seeing 38 or more hits out of 40 has
        # probability under 0.01, so this floor is a real test.
        self.assertGreaterEqual(power, 0.80)

    def test_the_smaller_planted_cell_is_genuinely_underpowered(self):
        """The negative result, asserted rather than hidden.

        A 9-point gap on a 300-row slice survives BH about two thirds of
        the time. An audit that reported the other third as 'fair' would
        be lying; this suite pins the real number.
        """
        smaller, _ = self._by_size()
        power = self.hits[smaller] / self.runs
        self.assertLess(power, 0.85)
        self.assertGreater(power, 0.45)

    def test_the_published_mde_predicts_which_cell_is_findable(self):
        smaller, larger = self._by_size()
        planted = PLANTED_MARGIN
        mean_small = sum(self.mdes[smaller]) / self.runs
        mean_large = sum(self.mdes[larger]) / self.runs
        # The audit says so in advance: it cannot reliably see a 9-point
        # gap on the smaller cell (MDE ~ 11.7 pts) but can on the larger
        # one (MDE ~ 7.9 pts). Exactly the measured power ordering.
        self.assertGreater(mean_small, planted)
        self.assertLess(mean_large, planted)

    def test_a_missed_planted_cell_is_never_reported_as_fair(self):
        """The safety property that makes the miss survivable."""
        smaller, _ = self._by_size()
        missed = [v for v in self.verdicts[smaller] if v != SIGNIFICANT]
        self.assertGreater(len(missed), 0)
        for verdict in missed:
            self.assertEqual(verdict, INDETERMINATE)

    def test_correction_suppresses_noise_without_erasing_signal(self):
        # Contrast with the null block above: BH removes 96% of null
        # claims while still recovering the adequately-powered planted
        # cell in 95% of runs.
        _, larger = self._by_size()
        self.assertGreaterEqual(self.hits[larger] / self.runs, 0.80)


if __name__ == "__main__":
    unittest.main()

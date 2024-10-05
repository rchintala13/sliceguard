"""Statistical primitives: Wilson vs Wald, Newcombe, BH, and the MDE."""

import random
import unittest
from math import comb

from src.stats import (
    benjamini_hochberg,
    minimum_detectable_disparity,
    newcombe_difference_interval,
    normal_cdf,
    normal_quantile,
    score_test_two_proportions,
    wald_interval,
    wilson_interval,
)


def exact_coverage(interval_fn, n, p, confidence=0.95):
    """Exact binomial coverage of an interval method -- no simulation.

    Sums the binomial pmf over every k whose interval contains p. This is
    the true coverage, not an estimate, so the assertions below have no
    Monte Carlo slack to argue about.
    """
    total = 0.0
    for k in range(n + 1):
        pmf = comb(n, k) * p ** k * (1 - p) ** (n - k)
        lo, hi = interval_fn(k, n, confidence)
        if lo <= p <= hi:
            total += pmf
    return total


class TestNormal(unittest.TestCase):
    def test_quantile_inverts_cdf(self):
        for p in (0.001, 0.025, 0.5, 0.975, 0.999):
            self.assertAlmostEqual(normal_cdf(normal_quantile(p)), p, places=10)

    def test_known_critical_value(self):
        self.assertAlmostEqual(normal_quantile(0.975), 1.959963985, places=8)


class TestWaldIsDegenerate(unittest.TestCase):
    """The motivating failure, stated as a test.

    If someone swaps wilson_interval for the normal approximation, these
    are the assertions that catch it.
    """

    def test_wald_collapses_to_zero_width_at_zero_successes(self):
        # 0 correct out of 20 is a perfectly ordinary tiny-slice outcome.
        wald_lo, wald_hi = wald_interval(0, 20)
        self.assertEqual(wald_hi - wald_lo, 0.0)
        # ... and so the Wald interval asserts, from 20 observations, that
        # the true accuracy is EXACTLY zero. It excludes p = 0.10, under
        # which seeing 0/20 has probability 0.12 -- a routine outcome.
        self.assertFalse(wald_lo <= 0.10 <= wald_hi)

        wilson_lo, wilson_hi = wilson_interval(0, 20)
        self.assertGreater(wilson_hi - wilson_lo, 0.0)
        self.assertTrue(wilson_lo <= 0.10 <= wilson_hi)
        self.assertAlmostEqual(comb(20, 0) * 0.9 ** 20, 0.12157665, places=6)

    def test_wald_escapes_the_unit_interval(self):
        lo, hi = wald_interval(19, 20)
        self.assertGreater(hi, 1.0)  # a probability above 1
        w_lo, w_hi = wilson_interval(19, 20)
        self.assertLessEqual(w_hi, 1.0)
        self.assertGreaterEqual(w_lo, 0.0)

    def test_wilson_holds_nominal_coverage_where_wald_does_not(self):
        cases = ((20, 0.90), (30, 0.95), (50, 0.86))
        for n, p in cases:
            wilson = exact_coverage(wilson_interval, n, p)
            wald = exact_coverage(wald_interval, n, p)
            with self.subTest(n=n, p=p):
                self.assertGreater(wilson, 0.93)
                # Wilson is closer to nominal in every one of these cases.
                self.assertLess(abs(wilson - 0.95), abs(wald - 0.95))
        # The worst case here: a 95% Wald interval that covers 78% of the
        # time. That is the number a fairness report would be built on.
        self.assertLess(exact_coverage(wald_interval, 30, 0.95), 0.80)

    def test_wilson_is_not_symmetric_about_the_point_estimate(self):
        # A shrunk, asymmetric interval is the signature of the score
        # inversion; a symmetric one means someone reinstated Wald.
        lo, hi = wilson_interval(18, 20)
        p_hat = 0.9
        self.assertNotAlmostEqual(p_hat - lo, hi - p_hat, places=3)
        self.assertGreater(p_hat - lo, hi - p_hat)


class TestScoreTestAndNewcombe(unittest.TestCase):
    def test_identical_proportions_give_p_one(self):
        self.assertAlmostEqual(
            score_test_two_proportions(50, 100, 100, 200), 1.0, places=12)

    def test_degenerate_groups_do_not_claim_certainty(self):
        # Everything correct in both arms: no evidence of a difference.
        self.assertEqual(score_test_two_proportions(20, 20, 40, 40), 1.0)

    def test_score_test_agrees_with_normal_theory(self):
        # Large, well-separated groups: the score z is computable by hand.
        p = score_test_two_proportions(800, 1000, 900, 1000)
        self.assertLess(p, 1e-9)

    def test_newcombe_interval_contains_the_point_estimate(self):
        lo, hi = newcombe_difference_interval(17, 20, 860, 1000)
        d = 17 / 20 - 0.86
        self.assertLess(lo, d)
        self.assertGreater(hi, d)
        self.assertGreaterEqual(lo, -1.0)
        self.assertLessEqual(hi, 1.0)

    def test_newcombe_stays_wide_when_an_arm_is_at_the_boundary(self):
        # 20/20 correct in the slice. The pooled-Wald difference interval
        # would have zero variance from this arm; Newcombe does not.
        lo, hi = newcombe_difference_interval(20, 20, 860, 1000)
        self.assertGreater(hi - lo, 0.10)
        self.assertTrue(lo < 0.0 < hi or lo > 0.0)

    def test_newcombe_never_claims_more_than_the_p_value(self):
        """Measured: Newcombe is conservative relative to the score test.

        Over 4000 random slice/reference pairs the two disagree 19% of the
        time, and EVERY disagreement is in the same direction -- the
        interval still straddles zero while the pooled score test is under
        0.05. The interval never claims significance the test denies, so
        ranking by the interval's lower bound can only under-claim harm,
        never over-claim it. (Expected: Newcombe inverts two separate
        score intervals; the p-value pools. They are close, not identical,
        and the audit is careful never to rely on them being identical.)
        """
        rng = random.Random(11)
        disagreements = 0
        runs = 4000
        for _ in range(runs):
            n1 = rng.randint(15, 400)
            n2 = rng.randint(500, 3000)
            k1 = sum(1 for _ in range(n1) if rng.random() < 0.80)
            k2 = sum(1 for _ in range(n2) if rng.random() < 0.86)
            lo, hi = newcombe_difference_interval(k1, n1, k2, n2)
            p = score_test_two_proportions(k1, n1, k2, n2)
            excludes_zero = lo > 0.0 or hi < 0.0
            if excludes_zero:
                # The safe direction, and the one the audit depends on.
                self.assertLess(p, 0.05)
            if p < 0.001:
                self.assertTrue(excludes_zero)
            if excludes_zero != (p < 0.05):
                disagreements += 1
        self.assertLess(disagreements / runs, 0.25)


class TestBenjaminiHochberg(unittest.TestCase):
    def test_step_up_sweep_is_present(self):
        # m=2, q=0.05. Rank 1 (p=0.026) FAILS its own 0.025 threshold, but
        # rank 2 (p=0.04) passes 0.05, and BH's step-up then rejects BOTH.
        # An implementation that compares each p to its own threshold and
        # stops rejects only one -- a real and common bug.
        rejected = benjamini_hochberg([0.026, 0.04], q=0.05)
        self.assertEqual(rejected, [True, True])

    def test_rejects_nothing_when_nothing_qualifies(self):
        self.assertEqual(benjamini_hochberg([0.4, 0.6, 0.9], q=0.05),
                         [False, False, False])

    def test_order_is_preserved_in_the_output(self):
        rejected = benjamini_hochberg([0.9, 0.0001, 0.5], q=0.05)
        self.assertEqual(rejected, [False, True, False])

    def test_is_strictly_more_powerful_than_bonferroni(self):
        m = 100
        p_values = [0.0001] * 10 + [0.5] * (m - 10)
        bh = benjamini_hochberg(p_values, q=0.05)
        bonferroni = [p <= 0.05 / m for p in p_values]
        self.assertEqual(sum(bh), 10)
        self.assertGreaterEqual(sum(bh), sum(bonferroni))

    def test_empty_family(self):
        self.assertEqual(benjamini_hochberg([], q=0.05), [])


class TestMinimumDetectableDisparity(unittest.TestCase):
    def test_mde_shrinks_as_the_slice_grows(self):
        sizes = [25, 100, 400, 1600]
        mdes = [minimum_detectable_disparity(n, 5000, 0.86, alpha=0.05)
                for n in sizes]
        self.assertEqual(mdes, sorted(mdes, reverse=True))

    def test_mde_is_not_optimistic(self):
        """The reported MDE must never overstate what the audit can find.

        Simulate at exactly the reported MDE and confirm the realised
        detection rate reaches the 0.80 design target. It comes out
        ABOVE target (the normal-approximation MDE is conservative),
        which is the safe direction: the report understates its own
        power rather than promising detection it cannot deliver.
        """
        rng = random.Random(7)
        n1, n2, p_ref, alpha, runs = 400, 5000, 0.86, 0.05, 600
        delta = minimum_detectable_disparity(n1, n2, p_ref, alpha=alpha)
        detected = 0
        for _ in range(runs):
            k1 = sum(1 for _ in range(n1) if rng.random() < p_ref - delta)
            k2 = sum(1 for _ in range(n2) if rng.random() < p_ref)
            if (score_test_two_proportions(k1, n1, k2, n2) < alpha
                    and k1 / n1 < k2 / n2):
                detected += 1
        # 3 binomial standard errors below the 0.80 design target:
        # sqrt(0.8 * 0.2 / 600) = 0.0163, so the floor is 0.80 - 0.049.
        self.assertGreaterEqual(detected / runs, 0.80 - 3 * 0.0163)

    def test_a_23_person_slice_can_barely_detect_anything(self):
        # The README's motivating slice. Even uncorrected, it needs a gap
        # of roughly 29 accuracy points before it would reliably show up.
        mde = minimum_detectable_disparity(23, 20000, 0.86, alpha=0.05)
        self.assertGreater(mde, 0.25)

    def test_a_20_person_slice_needs_a_catastrophe(self):
        # Under the family-corrected alpha a 20-row slice needs a gap of
        # 49 accuracy points. Measured, not asserted: simulating at that
        # gap detects it 97% of the time -- above the 80% target, because
        # the normal-approximation MDE is conservative at this n.
        mde = minimum_detectable_disparity(20, 20000, 0.86, alpha=0.05 / 200)
        self.assertGreater(mde, 0.45)
        rng = random.Random(3)
        detected = 0
        for _ in range(400):
            k1 = sum(1 for _ in range(20) if rng.random() < 0.86 - mde)
            k2 = sum(1 for _ in range(20000) if rng.random() < 0.86)
            if (score_test_two_proportions(k1, 20, k2, 20000) < 0.05 / 200
                    and k1 / 20 < k2 / 20000):
                detected += 1
        self.assertGreaterEqual(detected / 400, 0.80)

    def test_returns_none_when_nothing_is_detectable(self):
        # Both arms small AND the reference accuracy low: there is not
        # enough room below 10% for any gap to clear the corrected alpha,
        # so the honest answer is None -- "this audit could not have found
        # anything here", which the report renders as INDETERMINATE.
        self.assertIsNone(
            minimum_detectable_disparity(20, 40, 0.10, alpha=0.05 / 200))
        self.assertIsNone(minimum_detectable_disparity(0, 40, 0.86, alpha=0.05))


if __name__ == "__main__":
    unittest.main()

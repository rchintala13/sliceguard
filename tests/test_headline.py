"""THE HEADLINE TEST.

Two claims, both required:

  1. Ranking slices by raw point-estimate disparity -- what a generic
     fairness report prints -- puts tiny, noise-driven slices at the top.
  2. Ranking by the Newcombe lower bound after BH correction recovers the
     genuinely-affected slices instead.

Claim 1 alone is a complaint. Claim 2 alone is unconvincing. Measured
across many independent evaluation sets so neither is a lucky seed.
"""

import unittest

from src.audit import audit, naive_ranking, slice_predicate
from src.generator import generate_eval_set, true_disparity

# The organisation's action threshold: a gap of 5 accuracy points or more
# is worth investigating. It is a POLICY number, reused here to define
# ground truth so that "genuinely affected" means the same thing to the
# audit and to the scorer. It is not tuned to make the test pass.
MATERIAL = 0.05

RUNS = 40
TOP_K = 5


def _score_one(seed):
    records = generate_eval_set(6000, seed=seed)
    cache = {}

    def truth(sl):
        if sl.key not in cache:
            cache[sl.key] = true_disparity(records, slice_predicate(sl.key))
        return cache[sl.key]

    naive_top = [sl for _, sl in naive_ranking(records, max_order=3)[:TOP_K]]
    guarded = audit(records, max_order=3, material_disparity=MATERIAL)
    guarded_top = [r.slice for r in guarded[:TOP_K]]
    return {
        "naive_top1_is_noise": truth(naive_top[0]) < MATERIAL,
        "guarded_top1_is_noise": truth(guarded_top[0]) < MATERIAL,
        "naive_precision": sum(truth(s) >= MATERIAL for s in naive_top) / TOP_K,
        "guarded_precision": sum(truth(s) >= MATERIAL
                                 for s in guarded_top) / TOP_K,
        "naive_top1_n": naive_top[0].n,
        "guarded_top1_n": guarded_top[0].n,
    }


class TestNaiveRankingChasesNoise(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = [_score_one(1000 + i) for i in range(RUNS)]

    def _mean(self, key):
        return sum(r[key] for r in self.runs) / len(self.runs)

    def test_point_estimate_ranking_leads_with_a_noise_slice_about_half_the_time(self):
        rate = self._mean("naive_top1_is_noise")
        # Measured 0.525 over these 40 evaluation sets. Binomial standard
        # error at n=40 is at most 0.079, so 3 SE below the measured value
        # is 0.29 -- the floor asserted here. The claim being defended is
        # "this happens constantly", not "this happens exactly 52% of the
        # time".
        self.assertGreater(rate, 0.29)

    def test_guarded_ranking_leads_with_a_real_finding(self):
        rate = self._mean("guarded_top1_is_noise")
        # Measured 0.075. The guarded ranking is not perfect -- it can
        # still lead with a diluted parent slice whose true gap sits just
        # under the 5-point material line -- but it is an order of
        # magnitude better than the point-estimate ranking.
        self.assertLess(rate, self._mean("naive_top1_is_noise") / 3.0)

    def test_guarded_top_five_is_substantially_cleaner(self):
        naive = self._mean("naive_precision")
        guarded = self._mean("guarded_precision")
        # Measured: 0.565 -> 0.915 precision@5.
        self.assertGreater(guarded, naive + 0.20)
        self.assertGreater(guarded, 0.80)

    def test_the_noise_leader_is_the_small_slice(self):
        """The mechanism, not just the outcome.

        When the point-estimate ranking leads with a noise slice, that
        slice is small. This is what makes the failure systematic rather
        than bad luck: the largest observed gap is a maximum over ~200
        slices, and the sampling distribution of the maximum is dominated
        by whichever slices have the fattest tails, i.e. the smallest.
        """
        noisy = [r for r in self.runs if r["naive_top1_is_noise"]]
        self.assertGreater(len(noisy), 5)
        naive_ns = sorted(r["naive_top1_n"] for r in noisy)
        median_naive_n = naive_ns[len(naive_ns) // 2]
        guarded_ns = sorted(r["guarded_top1_n"] for r in self.runs)
        median_guarded_n = guarded_ns[len(guarded_ns) // 2]
        self.assertLess(median_naive_n, 60)
        self.assertGreater(median_guarded_n, 100)
        self.assertGreater(median_guarded_n, 2 * median_naive_n)


if __name__ == "__main__":
    unittest.main()

"""Audit mechanics: verdicts, the INDETERMINATE state, and MDE reporting."""

import unittest

from src.audit import (
    INDETERMINATE,
    MIN_SLICE_N,
    NO_MATERIAL,
    SIGNIFICANT,
    UNTESTABLE,
    audit,
    enumerate_slices,
    slice_predicate,
)
from src.generator import (
    ATTRIBUTES,
    PLANTED_CELLS,
    PLANTED_MARGIN,
    generate_eval_set,
    true_disparity,
)


class TestEnumeration(unittest.TestCase):
    def setUp(self):
        self.records = generate_eval_set(4000, seed=5)

    def test_enumerates_intersectional_slices(self):
        one_way = enumerate_slices(self.records, max_order=1)
        two_way = enumerate_slices(self.records, max_order=2)
        three_way = enumerate_slices(self.records, max_order=3)
        self.assertEqual(len(one_way), 4 + 4 + 3 + 2)
        self.assertGreater(len(two_way), len(one_way))
        # Hundreds of slices is the regime the project exists for.
        self.assertGreater(len(three_way), 150)

    def test_one_way_slices_of_an_attribute_partition_the_data(self):
        slices = enumerate_slices(self.records, max_order=1)
        for attr in ATTRIBUTES:
            total = sum(s.n for s in slices if s.key[0][0] == attr)
            self.assertEqual(total, len(self.records))

    def test_tiny_slices_are_enumerated_not_hidden(self):
        # The generator's 4% 'phone' channel is there to create these.
        tiny = [s for s in enumerate_slices(self.records, max_order=3)
                if 15 <= s.n <= 40]
        self.assertGreaterEqual(len(tiny), 5)


class TestPlantedDisparityIsFound(unittest.TestCase):
    def setUp(self):
        self.records = generate_eval_set(6000, seed=1)
        self.results = audit(self.records, max_order=3)

    def test_planted_cells_are_flagged(self):
        flagged = {r.slice.key for r in self.results
                   if r.verdict == SIGNIFICANT}
        for cell in PLANTED_CELLS:
            self.assertIn(cell, flagged, "planted cell %s not recovered" % (cell,))

    def test_flagged_slices_are_genuinely_affected(self):
        for r in self.results:
            if r.verdict == SIGNIFICANT:
                truth = true_disparity(self.records,
                                       slice_predicate(r.slice.key))
                # Every BH survivor has a real, positive true disparity --
                # no discovery is pointing the wrong way.
                self.assertGreater(truth, 0.0, r.slice.name)

    def test_planted_margin_is_recovered_within_the_interval(self):
        by_key = {r.slice.key: r for r in self.results}
        for cell in PLANTED_CELLS:
            r = by_key[cell]
            truth = true_disparity(self.records, slice_predicate(cell))
            self.assertLessEqual(r.interval[0], truth)
            self.assertGreaterEqual(r.interval[1], truth)
            # Slightly BELOW the planted margin, and necessarily so: the
            # reference group is "everyone else", which still contains the
            # other planted cell, so the baseline it is measured against is
            # itself depressed. An audit that expected to recover exactly
            # 9.0 points would be assuming a clean reference it never has.
            self.assertLess(truth, PLANTED_MARGIN)
            self.assertGreater(truth, 0.85 * PLANTED_MARGIN)


class TestSilenceIsNotAPass(unittest.TestCase):
    """A slice with too little data must never read as 'fair'."""

    def setUp(self):
        self.records = generate_eval_set(6000, seed=2)
        self.results = audit(self.records, max_order=3,
                             material_disparity=0.05)
        self.by_key = {r.slice.key: r for r in self.results}

    def test_no_slice_is_ever_labelled_fair_without_the_power_to_say_so(self):
        for r in self.results:
            if r.verdict == NO_MATERIAL:
                # The only route to a clean bill of health: the audit must
                # have been able to detect a material gap and did not.
                self.assertIsNotNone(r.mde)
                self.assertLessEqual(r.mde, 0.05)

    def test_tiny_slices_come_back_indeterminate_not_clean(self):
        tiny = [r for r in self.results if r.n <= 40]
        self.assertGreater(len(tiny), 0)
        for r in tiny:
            self.assertIn(r.verdict, (INDETERMINATE, UNTESTABLE, SIGNIFICANT))
            self.assertNotEqual(r.verdict, NO_MATERIAL)

    def test_below_gate_slices_make_no_claim_at_all(self):
        gated = [r for r in self.results if r.n < MIN_SLICE_N]
        for r in gated:
            self.assertEqual(r.verdict, UNTESTABLE)
            self.assertFalse(r.bh_rejected)
            self.assertIsNone(r.mde)

    def test_every_indeterminate_slice_says_what_it_could_not_have_found(self):
        indeterminate = [r for r in self.results if r.verdict == INDETERMINATE]
        self.assertGreater(len(indeterminate), 0)
        for r in indeterminate:
            self.assertTrue(r.notes)
            self.assertIn("pts", r.notes)
            if r.mde is not None:
                self.assertGreater(r.mde, 0.05)

    def test_mde_is_smaller_for_bigger_slices(self):
        tested = [r for r in self.results if r.mde is not None]
        smallest = min(tested, key=lambda r: r.n)
        largest = max(tested, key=lambda r: r.n)
        self.assertGreater(smallest.mde, largest.mde)

    def test_a_clean_slice_is_only_clean_when_it_is_big(self):
        clean = [r for r in self.results if r.verdict == NO_MATERIAL]
        self.assertGreater(len(clean), 0)
        # 5 accuracy points against an ~86% reference needs roughly a
        # thousand rows; nothing smaller may be called clean.
        self.assertGreater(min(r.n for r in clean), 500)


class TestNullDatasetProducesNoClaims(unittest.TestCase):
    def test_a_single_null_audit_typically_flags_nothing(self):
        # margin=0.0 removes the planted effect entirely.
        records = generate_eval_set(6000, seed=9, margin=0.0)
        results = audit(records, max_order=3)
        self.assertEqual([r.slice.name for r in results
                          if r.verdict == SIGNIFICANT], [])


if __name__ == "__main__":
    unittest.main()

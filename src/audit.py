"""Slice enumeration and the guarded audit itself.

The audit answers three questions per slice, and keeps them separate:

  1. How much worse is this slice than the rest of the population?
     -> point estimate, plus a Newcombe interval around it.
  2. Does that survive multiplicity correction across every slice tested?
     -> Benjamini-Hochberg at FDR q over the whole family.
  3. If it did not survive, could this slice ever have shown a problem?
     -> minimum detectable disparity. If the MDE is above what the
        organisation considers material, the verdict is INDETERMINATE,
        never "fair".

Point 3 is the one audits skip. Silence from an underpowered slice is not
a pass, and this module refuses to let it read as one.
"""

from dataclasses import dataclass, field
from itertools import combinations, product

from .generator import ATTRIBUTES
from .stats import (
    benjamini_hochberg,
    minimum_detectable_disparity,
    newcombe_difference_interval,
    score_test_two_proportions,
    wilson_interval,
)

SIGNIFICANT = "DISPARITY"
NO_MATERIAL = "NO_MATERIAL_DISPARITY"
INDETERMINATE = "INDETERMINATE"
UNTESTABLE = "INDETERMINATE_BELOW_GATE"

# A slice with fewer than this many rows has no usable interval at all
# (Wilson is defined but uselessly wide, and the reference comparison is
# numerically fragile). It is a validity floor, not a policy threshold --
# the policy work is done by the MDE, which subsumes any size gate.
MIN_SLICE_N = 10


@dataclass(frozen=True)
class Slice:
    key: tuple          # (("region", "south"), ("age_band", "60+"))
    n: int
    correct: int

    @property
    def name(self):
        return " & ".join("%s=%s" % (a, v) for a, v in self.key)

    @property
    def order(self):
        return len(self.key)


@dataclass
class SliceResult:
    slice: Slice
    n: int
    accuracy: float
    reference_n: int
    reference_accuracy: float
    disparity: float                 # reference minus slice; positive = worse
    wilson: tuple = (0.0, 1.0)
    interval: tuple = (-1.0, 1.0)    # Newcombe interval on the disparity
    p_value: float = 1.0
    bh_rejected: bool = False
    mde: object = None               # float, or None if nothing is detectable
    verdict: str = INDETERMINATE
    notes: str = ""

    @property
    def guarded_score(self):
        """Rank key: the lower confidence bound on the harm.

        Ranking by this instead of by `disparity` is the whole point. A
        tiny slice can post a huge point estimate, but its lower bound is
        pinned near (or below) zero because the data cannot rule out no
        effect at all.
        """
        return self.interval[0]


def enumerate_slices(records, max_order=2, min_n=MIN_SLICE_N,
                     attributes=ATTRIBUTES):
    """All 1..max_order-way attribute cells present in the data.

    Cells below `min_n` are returned too -- they must appear in the report
    as INDETERMINATE. Dropping them silently would recreate exactly the
    failure this project is about, just with the evidence hidden.
    """
    counts = {}
    for order in range(1, max_order + 1):
        for attrs in combinations(attributes, order):
            for r in records:
                key = tuple((a, r.value(a)) for a in attrs)
                n, c = counts.get(key, (0, 0))
                counts[key] = (n + 1, c + r.correct)
    out = [Slice(key=k, n=v[0], correct=v[1]) for k, v in counts.items()]
    out.sort(key=lambda s: (s.order, s.key))
    return out


def slice_predicate(key):
    return lambda r: all(r.value(a) == v for a, v in key)


def audit(records, max_order=2, q=0.05, confidence=0.95,
          material_disparity=0.05, min_n=MIN_SLICE_N, power=0.80):
    """Run the guarded audit. Returns results sorted worst-first.

    `material_disparity` is a POLICY input, not a statistical one: it is
    the smallest accuracy gap the organisation would act on. It decides
    only whether a non-significant slice is reported as "no material
    disparity found" or as "insufficient evidence". It never affects a
    p-value or an interval.
    """
    slices = enumerate_slices(records, max_order=max_order, min_n=min_n)
    total_n = len(records)
    total_correct = sum(r.correct for r in records)

    testable, results = [], []
    for sl in slices:
        ref_n = total_n - sl.n
        ref_c = total_correct - sl.correct
        if ref_n <= 0:
            continue
        acc = sl.correct / sl.n if sl.n else 0.0
        ref_acc = ref_c / ref_n
        res = SliceResult(
            slice=sl, n=sl.n, accuracy=acc,
            reference_n=ref_n, reference_accuracy=ref_acc,
            disparity=ref_acc - acc,
        )
        if sl.n < min_n:
            res.verdict = UNTESTABLE
            res.notes = "below validity floor of n=%d; not tested" % min_n
            results.append(res)
            continue
        res.wilson = wilson_interval(sl.correct, sl.n, confidence)
        lo, hi = newcombe_difference_interval(ref_c, ref_n, sl.correct, sl.n,
                                              confidence)
        res.interval = (lo, hi)
        res.p_value = score_test_two_proportions(sl.correct, sl.n, ref_c, ref_n)
        testable.append(res)
        results.append(res)

    # BH runs over the tested family only -- m is the number of tests
    # actually performed, which is what FDR control is defined against.
    rejects = benjamini_hochberg([r.p_value for r in testable], q=q)
    m = max(1, len(testable))
    for res, rej in zip(testable, rejects):
        # MDE is quoted at BH's most stringent possible threshold (q/m).
        # BH's actual cutoff depends on how many other slices reject, which
        # is unknowable when designing the audit, so this is the disparity
        # the audit could have caught no matter how the rest turned out.
        res.mde = minimum_detectable_disparity(
            res.n, res.reference_n, res.reference_accuracy,
            alpha=q / m, power=power)
        # Direction matters: a slice significantly BETTER than the
        # reference is not a fairness finding against that slice.
        res.bh_rejected = bool(rej) and res.disparity > 0
        if res.bh_rejected:
            res.verdict = SIGNIFICANT
        elif res.mde is not None and res.mde <= material_disparity:
            res.verdict = NO_MATERIAL
            res.notes = ("powered to detect %.1f pts; none found"
                         % (100 * res.mde))
        else:
            res.verdict = INDETERMINATE
            res.notes = ("insufficient evidence; smallest detectable gap "
                         + ("> %.0f pts" % (100 * res.reference_accuracy)
                            if res.mde is None
                            else "%.1f pts" % (100 * res.mde)))

    results.sort(key=lambda r: (-int(r.bh_rejected), -r.guarded_score))
    return results


def naive_ranking(records, max_order=2, min_n=MIN_SLICE_N):
    """What a generic fairness report prints: point estimates, sorted.

    No intervals, no multiplicity correction, no power statement. Kept in
    the library (not just the tests) because the headline test needs to
    compare against it, and because it is the honest baseline to show a
    reviewer.
    """
    slices = enumerate_slices(records, max_order=max_order, min_n=min_n)
    total_n = len(records)
    total_correct = sum(r.correct for r in records)
    rows = []
    for sl in slices:
        ref_n = total_n - sl.n
        if ref_n <= 0 or sl.n < min_n:
            continue
        acc = sl.correct / sl.n
        ref_acc = (total_correct - sl.correct) / ref_n
        rows.append((ref_acc - acc, sl))
    rows.sort(key=lambda t: -t[0])
    return rows

"""Small-sample-safe binomial statistics for slice auditing.

Everything here is stdlib-only. The three pieces that matter:

  * Wilson score intervals (NOT Wald) for a single proportion.
  * Newcombe's hybrid-score interval for a difference of proportions,
    which is built directly out of the two Wilson intervals.
  * Benjamini-Hochberg FDR control across the family of slices tested.

Why Wilson and not Wald: the Wald interval is p +/- z*sqrt(p(1-p)/n).
At k = 0 or k = n it collapses to a zero-width interval (a claim of
infinite certainty from 20 observations), and at moderate n near the
boundary it runs outside [0, 1]. A fairness audit spends most of its
time exactly there -- small slices with extreme observed rates -- so a
Wald-based audit is broken in the only regime it is asked about.
"""

import math

__all__ = [
    "normal_cdf",
    "normal_quantile",
    "wilson_interval",
    "wald_interval",
    "score_test_two_proportions",
    "newcombe_difference_interval",
    "benjamini_hochberg",
    "minimum_detectable_disparity",
]


def normal_cdf(x):
    """Standard normal CDF via erfc (exact to double precision, stdlib only)."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def normal_quantile(p):
    """Inverse standard normal CDF by bisection on normal_cdf.

    Bisection rather than a rational approximation (Acklam/Moro): we need
    this a handful of times per audit, not in an inner loop, and bisection
    to 1e-12 is trivially auditable -- no magic coefficients to get wrong.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("normal_quantile requires 0 < p < 1")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13:
            break
    return 0.5 * (lo + hi)


def wilson_interval(successes, n, confidence=0.95):
    """Wilson score interval for a binomial proportion.

    Solves |p_hat - p| = z * sqrt(p(1-p)/n) for p, i.e. it uses the
    standard error *under the hypothesised p* rather than under p_hat.
    That single change is what keeps the interval inside [0, 1] and
    non-degenerate at k = 0 and k = n.
    """
    if n <= 0:
        raise ValueError("wilson_interval requires n > 0")
    if not 0 <= successes <= n:
        raise ValueError("successes must lie in [0, n]")
    z = normal_quantile(1.0 - (1.0 - confidence) / 2.0)
    p_hat = successes / n
    denom = n + z * z
    centre = (successes + z * z / 2.0) / denom
    half = (z / denom) * math.sqrt(n * p_hat * (1.0 - p_hat) + z * z / 4.0)
    return (max(0.0, centre - half), min(1.0, centre + half))


def wald_interval(successes, n, confidence=0.95):
    """Normal-approximation (Wald) interval -- present ONLY as the foil.

    Deliberately NOT clamped to [0, 1]: the tests show it producing
    intervals that a clamp would hide. Never call this for a real audit.
    """
    if n <= 0:
        raise ValueError("wald_interval requires n > 0")
    z = normal_quantile(1.0 - (1.0 - confidence) / 2.0)
    p_hat = successes / n
    half = z * math.sqrt(p_hat * (1.0 - p_hat) / n)
    return (p_hat - half, p_hat + half)


def score_test_two_proportions(k1, n1, k2, n2):
    """Two-sided p-value for p1 == p2 (pooled-variance score test).

    The score test is the hypothesis test whose inverted form *is* the
    Wilson interval, so the p-values and the intervals in the report
    cannot disagree with each other.
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("score_test_two_proportions requires n1, n2 > 0")
    p_pool = (k1 + k2) / (n1 + n2)
    var = p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2)
    if var <= 0.0:
        # Every observation in both groups landed the same way; there is no
        # evidence of a difference, so the honest p-value is 1.
        return 1.0
    z = (k1 / n1 - k2 / n2) / math.sqrt(var)
    return math.erfc(abs(z) / math.sqrt(2.0))


def newcombe_difference_interval(k1, n1, k2, n2, confidence=0.95):
    """Newcombe hybrid-score interval for (p1 - p2).

    Built from the two Wilson intervals, so it inherits their behaviour at
    the boundary. The naive alternative -- difference +/- z * sqrt of the
    summed Wald variances -- reports a zero-width interval whenever either
    arm is at 0% or 100%, which is precisely the tiny-slice case.
    """
    l1, u1 = wilson_interval(k1, n1, confidence)
    l2, u2 = wilson_interval(k2, n2, confidence)
    d = k1 / n1 - k2 / n2
    lower = d - math.sqrt((d - (l1 - u2)) ** 2)
    upper = d + math.sqrt(((u1 - l2) - d) ** 2)
    # Written out the long way above for symmetry with the published form;
    # algebraically it is just (l1 - u2, u1 - l2).
    return (lower, upper)


def benjamini_hochberg(p_values, q=0.05):
    """Benjamini-Hochberg step-up procedure. Returns a list of bools.

    Reject H(i) for all i up to the largest i with p(i) <= i/m * q.
    Note the step-up sweep: rejecting the largest qualifying rank and
    everything below it is what gives BH its power advantage over
    Bonferroni. A per-test comparison without the sweep is a common and
    silently wrong reimplementation.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    cutoff_rank = 0
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= rank / m * q:
            cutoff_rank = rank
    rejected = [False] * m
    for rank, idx in enumerate(order, start=1):
        if rank <= cutoff_rank:
            rejected[idx] = True
    return rejected


def minimum_detectable_disparity(n_slice, n_reference, p_reference,
                                 alpha, power=0.80):
    """Smallest true disparity this slice could have detected, or None.

    Returns the delta (in proportion points, slice worse than reference by
    delta) at which a two-sided test at level `alpha` has at least `power`.
    None means "even a total collapse to 0% accuracy in this slice would
    not reliably have been flagged" -- the strongest possible statement of
    'this audit could not have told you'.

    `power` = 0.80 is the conventional design target and is exposed as a
    parameter rather than baked in, because it is a policy choice about how
    much missed harm the auditor will tolerate, not a statistical fact.
    """
    if n_slice <= 0 or n_reference <= 0:
        return None
    z_alpha = normal_quantile(1.0 - alpha / 2.0)

    def achieved_power(delta):
        p1 = max(0.0, p_reference - delta)
        se = math.sqrt(p1 * (1.0 - p1) / n_slice
                       + p_reference * (1.0 - p_reference) / n_reference)
        if se <= 0.0:
            return 1.0
        return (normal_cdf(delta / se - z_alpha)
                + normal_cdf(-delta / se - z_alpha))

    if achieved_power(p_reference) < power:
        return None
    lo, hi = 0.0, p_reference
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if achieved_power(mid) < power:
            lo = mid
        else:
            hi = mid
    return hi

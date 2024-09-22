"""The 60-second artefact: the same audit, printed twice.

Once the way a generic fairness report prints it (point estimates, sorted)
and once guarded (Wilson/Newcombe intervals, BH correction, MDE). ASCII
only -- the target console is Windows cp1252.
"""

from .audit import (
    INDETERMINATE,
    NO_MATERIAL,
    SIGNIFICANT,
    UNTESTABLE,
    audit,
    naive_ranking,
    slice_predicate,
)
from .generator import PLANTED_CELLS, PLANTED_MARGIN, generate_eval_set, true_disparity

SEED = 1
N_RECORDS = 6000
MATERIAL = 0.05
NULL_RUNS = 100
RULE = "-" * 96


def _pct(x):
    return "%+5.1f" % (100.0 * x)


def _truth_tag(records, key):
    truth = true_disparity(records, slice_predicate(key))
    if truth >= MATERIAL:
        return "REAL  %s" % _pct(truth)
    return "noise %s" % _pct(truth)


def header(title):
    print("")
    print("=" * 96)
    print(title)
    print("=" * 96)


def main():
    records = generate_eval_set(N_RECORDS, seed=SEED)
    overall = sum(r.correct for r in records) / len(records)

    header("SLICEGUARD -- fairness slice audit of an automated underwriting model")
    print("Evaluation rows      : %d" % len(records))
    print("Overall accuracy     : %.1f%%" % (100 * overall))
    print("Planted ground truth : %d intersectional cells are genuinely %.0f pts worse;"
          % (len(PLANTED_CELLS), 100 * PLANTED_MARGIN))
    for cell in PLANTED_CELLS:
        print("                       - %s"
              % " & ".join("%s=%s" % kv for kv in cell))
    print("                       every other slice differs by sampling noise only.")

    naive = naive_ranking(records, max_order=3)
    results = audit(records, max_order=3, material_disparity=MATERIAL)

    header("1. THE GENERIC REPORT -- rank slices by the raw accuracy gap")
    print("%-46s %6s %8s   %s" % ("slice", "n", "gap", "ground truth"))
    print(RULE)
    for gap, sl in naive[:6]:
        print("%-46s %6d %7s    %s"
              % (sl.name, sl.n, _pct(gap), _truth_tag(records, sl.key)))
    print(RULE)
    print("The top of this list is a %d-person slice. Its gap is noise: the largest"
          % naive[0][1].n)
    print("observed gap across %d slices is a maximum over %d noisy estimates, and the"
          % (len(naive), len(naive)))
    print("smallest slices have the fattest tails, so they win it almost every time.")
    print("This is the three-week investigation that finds nothing.")

    header("2. THE GUARDED REPORT -- Wilson/Newcombe intervals + Benjamini-Hochberg")
    print("%-46s %6s %8s %17s %7s  %s"
          % ("slice", "n", "gap", "95% CI on gap", "MDE", "verdict"))
    print(RULE)
    for r in results:
        if r.verdict != SIGNIFICANT:
            continue
        print("%-46s %6d %7s   [%s,%s] %6s  %s"
              % (r.slice.name, r.n, _pct(r.disparity),
                 _pct(r.interval[0]), _pct(r.interval[1]),
                 "%.1f" % (100 * r.mde), _truth_tag(records, r.slice.key)))
    print(RULE)
    print("Ranked by the LOWER bound of the interval, not the point estimate:")
    print("the question a business asks is 'how bad is it at least', not 'what")
    print("number did we happen to observe'.")

    header("3. WHAT THE AUDIT COULD NOT HAVE FOUND -- the loud tiny slices")
    print("%-46s %6s %8s %17s %7s  %s"
          % ("slice", "n", "gap", "95% CI on gap", "MDE", "verdict"))
    print(RULE)
    shown = 0
    for gap, sl in naive:
        r = next(res for res in results if res.slice.key == sl.key)
        if r.n > 45 or r.verdict == SIGNIFICANT:
            continue
        print("%-46s %6d %7s   [%s,%s] %6s  %s"
              % (r.slice.name, r.n, _pct(r.disparity),
                 _pct(r.interval[0]), _pct(r.interval[1]),
                 "n/a" if r.mde is None else "%.1f" % (100 * r.mde),
                 r.verdict))
        shown += 1
        if shown == 5:
            break
    print(RULE)
    print("INDETERMINATE, not 'fair'. The 19-person slice would need a gap of roughly 49")
    print("points before this audit could reliably have seen it, so its silence")
    print("carries no information. Silence is not a pass.")

    header("4. VERDICT COUNTS")
    labels = ((SIGNIFICANT, "disparity survives BH correction"),
              (NO_MATERIAL, "powered to rule out a 5-pt gap, none found"),
              (INDETERMINATE, "insufficient evidence either way"),
              (UNTESTABLE, "below the n=10 validity floor, not tested"))
    for verdict, blurb in labels:
        print("  %-28s %4d   %s"
              % (verdict, sum(1 for r in results if r.verdict == verdict), blurb))

    header("5. DOES THE CORRECTION ACTUALLY DO ANYTHING? (%d null datasets)" % NULL_RUNS)
    print("Same generator with the planted margin set to zero, so EVERY claim is false.")
    raw_runs = bh_runs = raw_total = bh_total = 0
    for i in range(NULL_RUNS):
        null_records = generate_eval_set(3000, seed=90000 + i, margin=0.0)
        null_results = audit(null_records, max_order=2)
        raw = sum(1 for r in null_results if r.verdict != UNTESTABLE
                  and r.p_value < 0.05 and r.disparity > 0)
        bh = sum(1 for r in null_results if r.verdict == SIGNIFICANT)
        raw_total += raw
        bh_total += bh
        raw_runs += raw > 0
        bh_runs += bh > 0
    print("")
    print("  %-34s %12s %12s" % ("", "uncorrected", "BH at q=0.05"))
    print("  " + RULE[:60])
    print("  %-34s %11.0f%% %11.0f%%"
          % ("null audits raising an alarm", 100.0 * raw_runs / NULL_RUNS,
             100.0 * bh_runs / NULL_RUNS))
    print("  %-34s %12.2f %12.2f"
          % ("false claims per audit", raw_total / NULL_RUNS, bh_total / NULL_RUNS))
    print("")
    print("Every one of those is a fairness crisis reported on data containing no")
    print("unfairness at all.")
    print("")


if __name__ == "__main__":
    main()

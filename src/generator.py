"""Synthetic model-evaluation set with a PLANTED, exactly-known disparity.

The generator produces underwriting decisions for applicants described by
four categorical attributes. Slice auditing then enumerates 1-way and
2-way (intersectional) slices over those attributes.

Three things are deliberately built in:

  1. A small, known set of intersectional cells whose true accuracy is
     depressed by a fixed margin. These are the findings an audit SHOULD
     surface.
  2. A rare `channel` level that makes many intersectional cells tiny
     (n roughly 15-40) with *no* true effect at all. These are the
     false-alarm factory that a point-estimate ranking walks straight
     into.
  3. Exact ground truth: because the generator knows each record's true
     success probability, the true disparity of ANY slice can be computed
     analytically from the realised sample rather than guessed. Dilution
     (a 1-way parent slice that partly contains an affected cell) is
     therefore handled correctly instead of being hand-waved.
"""

import random
from dataclasses import dataclass

REGIONS = ("north", "south", "east", "west")
AGE_BANDS = ("18-29", "30-44", "45-59", "60+")
CHANNELS = ("web", "broker", "phone")
PRIORS = ("new", "renewal")

ATTRIBUTES = ("region", "age_band", "channel", "prior")

# The channel mix is deliberately lopsided: 'phone' is a 4% channel, so
# every 2-way cell containing it lands in the n = 15-40 range that breaks
# point-estimate auditing. This is the realistic shape of the problem, not
# an adversarial contrivance -- rare intersections are the norm.
CHANNEL_WEIGHTS = (0.62, 0.34, 0.04)
REGION_WEIGHTS = (0.34, 0.31, 0.24, 0.11)
AGE_WEIGHTS = (0.22, 0.33, 0.28, 0.17)
PRIOR_WEIGHTS = (0.42, 0.58)

BASE_ACCURACY = 0.86

# The planted harm: two intersectional cells where the model is genuinely
# worse. Both are large enough to be detectable, which is the point -- the
# audit's job is to find these rather than the loud tiny ones.
PLANTED_CELLS = (
    (("region", "south"), ("age_band", "60+")),
    (("region", "east"), ("prior", "new")),
)
PLANTED_MARGIN = 0.09


@dataclass(frozen=True)
class Record:
    """One evaluation-set row: attributes, the model's outcome, and truth."""
    region: str
    age_band: str
    channel: str
    prior: str
    correct: int
    true_p: float

    def value(self, attribute):
        return getattr(self, attribute)


def _pick(rng, values, weights):
    return rng.choices(values, weights=weights, k=1)[0]


def _matches(record_attrs, cell):
    return all(record_attrs[attr] == val for attr, val in cell)


def true_probability(attrs, margin=PLANTED_MARGIN, base=BASE_ACCURACY):
    """The model's true per-record accuracy. Only PLANTED_CELLS differ."""
    p = base
    for cell in PLANTED_CELLS:
        if _matches(attrs, cell):
            p -= margin
    return p


def generate_eval_set(n_records=6000, seed=0, margin=PLANTED_MARGIN,
                      base=BASE_ACCURACY):
    """Draw an evaluation set. `margin = 0.0` gives a pure-null dataset."""
    rng = random.Random(seed)
    records = []
    for _ in range(n_records):
        attrs = {
            "region": _pick(rng, REGIONS, REGION_WEIGHTS),
            "age_band": _pick(rng, AGE_BANDS, AGE_WEIGHTS),
            "channel": _pick(rng, CHANNELS, CHANNEL_WEIGHTS),
            "prior": _pick(rng, PRIORS, PRIOR_WEIGHTS),
        }
        p = true_probability(attrs, margin=margin, base=base)
        records.append(Record(correct=1 if rng.random() < p else 0,
                              true_p=p, **attrs))
    return records


def true_disparity(records, predicate):
    """Exact true disparity (reference minus slice) for an arbitrary slice.

    Averages the generator's true per-record probabilities over the slice
    and over its complement. This is truth *conditional on the realised
    covariates*, which is the quantity a slice audit is actually estimating
    -- so a discovery can be scored right or wrong with no ambiguity.
    """
    in_p, out_p = [], []
    for r in records:
        (in_p if predicate(r) else out_p).append(r.true_p)
    if not in_p or not out_p:
        return None
    return sum(out_p) / len(out_p) - sum(in_p) / len(in_p)

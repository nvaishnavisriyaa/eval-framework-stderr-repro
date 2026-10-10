"""The three defects as tests.

Run against the shipped code and they fail; against the fix they pass::

    pytest test_stderr.py -v                 # the fix
    TARGET=current pytest test_stderr.py -v  # the shipped code

The ``TARGET`` switch exists so the same assertions document both sides.
"""

import math
import os
import random

import numpy as np
import pytest

from result import Result

if os.environ.get("TARGET") == "current":
    from aggregate_current import aggregate_results
else:
    from aggregate_fixed import aggregate_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def accuracy_rows(n_subjects, n_per_subject, p=0.6, seed=1, last_subject_n=None):
    rnd = random.Random(seed)
    rows, per_subject = [], {}
    for s in range(n_subjects):
        subject = f"subject_{s:02d}"
        n = n_per_subject if (last_subject_n is None or s < n_subjects - 1) else last_subject_n
        values = [float(rnd.random() < p) for _ in range(n)]
        per_subject[subject] = values
        rows += [Result("Accuracy Completion", subject, "", v) for v in values]
    return rows, per_subject


def textbook_standard_error(per_subject):
    """Standard error of an unweighted mean of group means."""
    subjects = len(per_subject)
    total = sum(float(np.std(v, ddof=1)) ** 2 / len(v) for v in per_subject.values())
    return math.sqrt(total) / subjects


# ---------------------------------------------------------------------------
# Defect 1: inflation by sqrt(number of subjects)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_subjects", [1, 2, 4, 9, 14, 57])
def test_standard_error_matches_textbook_value(n_subjects):
    """StdErr must be the standard error of the score that is reported next to it."""
    rows, per_subject = accuracy_rows(n_subjects, 100)
    got = aggregate_results(rows)["StdErr Accuracy Completion"]
    assert got == pytest.approx(textbook_standard_error(per_subject), rel=1e-9)


def test_standard_error_shrinks_as_subjects_are_added():
    """More data must narrow the interval. The shipped code leaves it flat."""
    errors = []
    for n_subjects in (2, 8, 32):
        rows, _ = accuracy_rows(n_subjects, 100, seed=5)
        errors.append(aggregate_results(rows)["StdErr Accuracy Completion"])
    assert errors[0] > errors[1] > errors[2], f"standard error did not shrink: {errors}"


# ---------------------------------------------------------------------------
# Defect 2: samples counted once per dimension
# ---------------------------------------------------------------------------


def test_num_samples_counts_each_value_once():
    """A metric with several keys and several subjects must not double count."""
    rows = []
    for subject in ("algebra", "geometry", "counting"):
        for key in ("Stop", "Length", "Repetition"):
            for value in (1.0, 0.0, 1.0, 0.0):
                rows.append(Result("FinishReason", subject, key, value))
    got = aggregate_results(rows)
    assert got["NumSamples FinishReason"] == len(rows)


def test_num_samples_is_independent_of_subject_labelling():
    """Relabelling the same samples across subjects must not change the count."""
    values = [float(i % 3 == 0) for i in range(36)]
    one_subject = [Result("FinishReason", "default", k, v) for k in ("a", "b") for v in values]
    many_subjects = [
        Result("FinishReason", f"subject_{i % 3}", k, v) for k in ("a", "b") for i, v in enumerate(values)
    ]
    assert (
        aggregate_results(one_subject)["NumSamples FinishReason"]
        == aggregate_results(many_subjects)["NumSamples FinishReason"]
    )


# ---------------------------------------------------------------------------
# Defect 3: one single-sample group voids the whole figure
# ---------------------------------------------------------------------------


def test_one_single_sample_subject_does_not_void_the_standard_error():
    rows, _ = accuracy_rows(57, 100, seed=3, last_subject_n=1)
    got = aggregate_results(rows)["StdErr Accuracy Completion"]
    assert not math.isnan(got), "5601 samples across 57 subjects produced no standard error"
    assert got > 0.0


def test_standard_error_is_nan_only_when_nothing_can_be_estimated():
    """One sample in total genuinely has no standard error, and should say so."""
    rows = [Result("Accuracy Completion", "only", "", 1.0)]
    assert math.isnan(aggregate_results(rows)["StdErr Accuracy Completion"])


# ---------------------------------------------------------------------------
# The scores themselves must not move
# ---------------------------------------------------------------------------


def test_reported_scores_are_untouched_by_the_fix():
    from aggregate_current import aggregate_results as current
    from aggregate_fixed import aggregate_results as fixed

    rows, _ = accuracy_rows(9, 50, seed=11, last_subject_n=1)
    before, after = current(rows), fixed(rows)
    assert set(before) == set(after)
    for name in before:
        if name.split()[0] in {"StdErr", "NumSamples"}:
            continue
        assert after[name] == pytest.approx(before[name], rel=1e-12, nan_ok=True), name

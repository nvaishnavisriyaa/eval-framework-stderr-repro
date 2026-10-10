"""Reproduces three standard-error defects in eval-framework v0.15.2 and checks the fix.

Run with:  python repro.py
"""

import math
import random

import numpy as np

from aggregate_current import aggregate_results as current
from aggregate_fixed import aggregate_results as fixed
from result import Result

RULE = "=" * 78


def heading(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------


def benchmark(n_subjects: int, n_per_subject: int, p: float = 0.6, seed: int = 1, last_subject_n: int | None = None):
    """A single-key accuracy metric spread over ``n_subjects`` subjects.

    This is the shape of every multi-subject benchmark: MMLU is 57 subjects,
    MMLU-Pro 14, MATH 7.
    """
    rnd = random.Random(seed)
    rows: list[Result] = []
    per_subject: dict[str, list[float]] = {}
    for s in range(n_subjects):
        subject = f"subject_{s:02d}"
        n = n_per_subject if (last_subject_n is None or s < n_subjects - 1) else last_subject_n
        values = [float(rnd.random() < p) for _ in range(n)]
        per_subject[subject] = values
        rows += [Result("Accuracy Completion", subject, "", v) for v in values]
    return rows, per_subject


def analytic_standard_error(per_subject: dict[str, list[float]]) -> float:
    """The textbook standard error of the reported score.

    The score is the unweighted mean of the per-subject means, so
        Var(score) = (1 / S^2) * sum_i Var(mean_i),  Var(mean_i) = s_i^2 / n_i
    """
    subjects = len(per_subject)
    total = sum(float(np.std(v, ddof=1)) ** 2 / len(v) for v in per_subject.values())
    return math.sqrt(total) / subjects


# ---------------------------------------------------------------------------
# 0. Fidelity: the extracted copy still matches the upstream test suite
# ---------------------------------------------------------------------------

UPSTREAM_FIXTURE = [
    # (subject, metric, key, value, error) - copied from
    # tests/tests_eval_framework/test_evaluation_generator.py::test_aggregate_results
    ("subject1", "metric1", "key1", 5.0, None),
    ("subject1", "metric1", "key1", 2.0, None),
    ("subject1", "metric1", "key1", None, "error"),
    ("subject1", "metric1", "key1", 2.0, None),
    ("subject1", "metric1", "key2", 0.5, None),
    ("subject1", "metric1", "key2", 1.5, None),
    ("subject1", "metric1", "key2", None, None),
    ("subject1", "metric2", "key", 3.0, None),
    ("subject1", "metric2", "key", 3.0, None),
    ("subject1", "metric3", None, 20.0, None),
    ("subject2", "metric1", "key1", 4.0, None),
    ("subject2", "metric1", "key2", 0.0, None),
    ("subject2", "metric1", "key2", None, "error"),
    ("subject2", "metric2", "key", None, None),
    ("subject2", "metric3", None, 18.0, None),
]

# The values that test asserts today.
UPSTREAM_ASSERTIONS = {
    "Average metric1": 2.0,
    "Average metric1 (micro)": 15 / 7,
    "Average metric1 - key1": 3.5,
    "Average metric1 - key2": 0.5,
    "ErrorFreeRatio metric1": 0.8,
    "StdErr metric1 - key1": 0.8660254037844386,
    "StdErr metric1 - key2": 0.3535533905932738,
    "StdErr metric1 - subject1": 0.4978909578906802,
    "NumSamples metric1 - key1": 4,
    "NumSamples metric1 - key2": 4,
    "NumSamples metric1 - subject1": 6,
    "NumSamples metric1 - subject2": 2,
    "NumSamples metric1": 16,
}


def fixture_results() -> list[Result]:
    return [Result(m, s, k, v, e) for s, m, k, v, e in UPSTREAM_FIXTURE]


def check_fidelity() -> None:
    heading("0. The extracted copy is faithful to upstream")
    got = current(fixture_results())
    worst = 0
    for name, expected in UPSTREAM_ASSERTIONS.items():
        actual = got[name]
        ok = (isinstance(actual, float) and math.isnan(actual) and math.isnan(float(expected))) or math.isclose(
            float(actual), float(expected), rel_tol=1e-12
        )
        worst += 0 if ok else 1
        print(f"  {'ok ' if ok else 'FAIL'} {name:<34} expected {expected!r:<22} got {actual!r}")
    assert math.isnan(got["StdErr metric1"]), "upstream asserts StdErr metric1 is nan"
    print(f"\n  {'ok ':<4} {'StdErr metric1':<34} expected nan                   got nan")
    print(f"\n  {worst} mismatches. The copy in aggregate_current.py reproduces the upstream")
    print("  test assertions, so the behaviour shown below is the shipped behaviour.")


# ---------------------------------------------------------------------------
# 1. StdErr is too large by sqrt(number of subjects)
# ---------------------------------------------------------------------------


def check_inflation() -> None:
    heading("1. StdErr is too large by exactly sqrt(number of subjects)")
    print("\n  The reported score is a macro average over subjects. Its standard error")
    print("  is sqrt(sum of per-subject variances) / S. The code divides by S instead")
    print("  of taking the root of the sum, so the result is sqrt(S) too large.\n")
    header = f"  {'subjects':>8} {'n/subj':>7} {'current':>11} {'fixed':>11} {'textbook':>11} {'cur/true':>9} {'sqrt(S)':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for subjects, n in [(1, 200), (2, 200), (4, 200), (9, 200), (14, 150), (57, 100)]:
        rows, per_subject = benchmark(subjects, n)
        cur = current(rows)["StdErr Accuracy Completion"]
        fix = fixed(rows)["StdErr Accuracy Completion"]
        true = analytic_standard_error(per_subject)
        print(
            f"  {subjects:>8} {n:>7} {cur:>11.6f} {fix:>11.6f} {true:>11.6f} "
            f"{cur / true:>9.3f} {math.sqrt(subjects):>8.3f}"
        )
    print("\n  The current column barely moves as subjects are added. A standard error has")
    print("  to shrink as data is added, which is the signature of the bug. The fixed")
    print("  column matches the textbook value to every digit shown, including at S=1,")
    print("  where the current code is already correct.")


def check_decision_impact() -> None:
    heading("1b. What that costs you: the smallest gap you could call significant")
    rows, per_subject = benchmark(57, 100)
    cur = current(rows)["StdErr Accuracy Completion"]
    fix = fixed(rows)["StdErr Accuracy Completion"]
    print("\n  MMLU shape, 57 subjects of 100 samples. Two models compared at 95%:\n")
    for label, se in (("as reported today", cur), ("with the fix     ", fix)):
        print(f"    {label}: StdErr {se:.4f} -> needs a {1.96 * math.sqrt(2) * se * 100:5.2f} point gap")
    print("\n  Verdict by gap size:\n")
    print(f"    {'gap':>6}  {'as reported today':<20} {'with the fix':<20}")
    print("    " + "-" * 46)
    for points in (1, 2, 3, 5, 8, 13, 14):
        gap = points / 100

        def verdict(se: float) -> str:
            return "significant" if gap / (math.sqrt(2) * se) > 1.96 else "no difference"

        flag = "  <- disagree" if verdict(cur) != verdict(fix) else ""
        print(f"    {points:>4}pt  {verdict(cur):<20} {verdict(fix):<20}{flag}")
    print("\n  Real model gaps live in the 2 to 8 point range. Every one of them is")
    print("  currently reported as no difference.")


# ---------------------------------------------------------------------------
# 2. Samples counted twice
# ---------------------------------------------------------------------------


def check_double_count() -> None:
    heading("2. Samples are counted twice when a metric has keys and subjects")
    results = fixture_results()
    cur = current(results)
    fix = fixed(results)
    metric1 = [r for r in results if r.metric_name == "metric1"]
    error_free = [r for r in metric1 if r.error is None]
    real_values = [r for r in error_free if r.value is not None]

    print(f"\n  Upstream fixture, metric1: {len(metric1)} rows, {len(error_free)} error-free,")
    print(f"  {len(real_values)} of which carry a value. The upstream test already says the")
    print('  micro average is "the plain mean of the 7 error-free values".\n')
    print(f"    per-key   NumSamples: {cur['NumSamples metric1 - key1']} + {cur['NumSamples metric1 - key2']}"
          f" = {cur['NumSamples metric1 - key1'] + cur['NumSamples metric1 - key2']}")
    print(f"    per-subj  NumSamples: {cur['NumSamples metric1 - subject1']} + {cur['NumSamples metric1 - subject2']}"
          f" = {cur['NumSamples metric1 - subject1'] + cur['NumSamples metric1 - subject2']}")
    print(f"\n    NumSamples metric1 (current) = {cur['NumSamples metric1']!r}   <- both dimensions added together")
    print(f"    NumSamples metric1 (fixed)   = {fix['NumSamples metric1']!r}   <- the values actually averaged")
    print("\n  The breakdown loop walks the same samples once per dimension, so the")
    print("  accumulators inside it see everything twice. Upstream asserts the doubled")
    print("  value, which is how it stayed. Single-key metrics are unaffected:")
    print(f"    NumSamples metric2: current {cur['NumSamples metric2']!r}, fixed {fix['NumSamples metric2']!r}")
    print(f"    NumSamples metric3: current {cur['NumSamples metric3']!r}, fixed {fix['NumSamples metric3']!r}")
    print("\n  Live on every multi-subject completion benchmark, because FinishReason")
    print("  emits 3 keys and tasks/base.py attaches it to every completion task.")


# ---------------------------------------------------------------------------
# 3. One single-sample group voids the whole figure
# ---------------------------------------------------------------------------


def check_nan_propagation() -> None:
    heading("3. One subject with a single sample makes the whole StdErr nan")
    healthy_rows, healthy_subjects = benchmark(57, 100, seed=3)
    short_rows, short_subjects = benchmark(57, 100, seed=3, last_subject_n=1)

    print("\n  57 subjects of 100 samples:")
    print(f"    current {current(healthy_rows)['StdErr Accuracy Completion']:.6f}"
          f"   fixed {fixed(healthy_rows)['StdErr Accuracy Completion']:.6f}")
    print("\n  The same run with one subject reduced to a single sample:")
    cur_short = current(short_rows)["StdErr Accuracy Completion"]
    fix_short = fixed(short_rows)["StdErr Accuracy Completion"]
    print(f"    current {cur_short!r}   fixed {fix_short:.6f}")
    print(f"\n    score is still fine: {current(short_rows)['Average Accuracy Completion']:.6f}")
    print("\n  pandas std uses ddof=1, so a group of one gives nan. The per-group write is")
    print("  guarded by `if math.isnan(std)`, but the accumulator line sits outside that")
    print("  guard, so the nan reaches the combined figure. 5601 good samples produce no")
    print("  usable standard error at all.")


# ---------------------------------------------------------------------------
# 4. The fix moves no score
# ---------------------------------------------------------------------------


def check_scores_unchanged() -> None:
    heading("4. The fix changes no reported score")
    cases = {
        "upstream fixture": fixture_results(),
        "57 subjects": benchmark(57, 100, seed=7)[0],
        "1 subject": benchmark(1, 300, seed=8)[0],
        "one singleton subject": benchmark(9, 50, seed=9, last_subject_n=1)[0],
    }
    touched = {"StdErr", "NumSamples"}
    all_ok = True
    for label, rows in cases.items():
        cur, fix = current(rows), fixed(rows)
        score_keys = [k for k in cur if k.split()[0] not in touched]
        assert set(cur) == set(fix), f"{label}: key sets differ"
        bad = []
        for k in score_keys:
            a, b = cur[k], fix[k]
            same = (
                a == b
                or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))
                or (isinstance(a, float) and isinstance(b, float) and math.isclose(a, b, rel_tol=1e-12))
            )
            if not same:
                bad.append(k)
        all_ok &= not bad
        print(f"  {'ok ' if not bad else 'FAIL'} {label:<24} {len(score_keys):>3} score keys identical"
              + (f"  differs: {bad}" if bad else ""))
    print(f"\n  Every Average, Average (micro), Average (including Errors) and")
    print("  ErrorFreeRatio value is untouched. Only StdErr and NumSamples change.")
    assert all_ok


if __name__ == "__main__":
    check_fidelity()
    check_inflation()
    check_decision_impact()
    check_double_count()
    check_nan_propagation()
    check_scores_unchanged()
    print(f"\n{RULE}\nAll checks ran.\n{RULE}")

# Three standard-error defects in eval-framework

Reproduction and fix for three bugs in the result aggregation of
[Aleph-Alpha-Research/eval-framework](https://github.com/Aleph-Alpha-Research/eval-framework),
all in `src/eval_framework/evaluation_generator.py`, method
`EvaluationGenerator._aggregate_results`.

Checked against **v0.15.2**, commit `8c570ad`.

Reported upstream as
[eval-framework#1037](https://github.com/Aleph-Alpha-Research/eval-framework/issues/1037).

The headline: **on every multi-subject benchmark, the reported standard error is
too large by a factor of `sqrt(number of subjects)`.** For MMLU that is 7.55x. The
error bars are wide enough to hide any model difference a team would plausibly be
arguing about.

```
$ python repro.py

  subjects  n/subj     current       fixed    textbook  cur/true  sqrt(S)
  -----------------------------------------------------------------------
         1     200    0.034026    0.034026    0.034026     1.000    1.000
         2     200    0.034302    0.024255    0.024255     1.414    1.414
         4     200    0.034760    0.017380    0.017380     2.000    2.000
         9     200    0.034685    0.011562    0.011562     3.000    3.000
        14     150    0.039940    0.010674    0.010674     3.742    3.742
        57     100    0.048974    0.006487    0.006487     7.550    7.550
```

Look at the `current` column. It barely moves as subjects are added. A standard
error has to shrink as data is added, so that flat column is the bug in one
glance. Single-subject benchmarks are already correct, which is why this went
unnoticed.

## Running it

```bash
pip install -r requirements.txt
python repro.py                        # all three defects, with the evidence
pytest test_stderr.py -v               # 12 pass against the fix
TARGET=current pytest test_stderr.py   # 9 fail against the shipped code
```

No GPU, no model and no dataset needed. The aggregation is pure pandas.

## The three defects

### 1. The combined standard error is inflated by `sqrt(number of groups)`

The reported score is an unweighted mean of per-group means, so its standard
error is

```
StdErr = sqrt( sum_groups Var(group mean) ) / number_of_groups
```

The code takes the root of the whole thing instead:

```python
aggregated_results[f"StdErr {metric}"] = np.sqrt(
    std_err_mean_sum_of_squares / std_err_mean_num_subjects
)
```

`sqrt(sum / n)` instead of `sqrt(sum) / n`, which is `sqrt(n)` too large. The
ratio measured above is exactly `sqrt(S)` at every size tested, and exactly 1.0
at `S = 1`.

The comment above that line already disagrees with the code it describes:

```python
# SEM = sqrt(sum(variance_i * n_i) / i)
```

The comment multiplies by `n_i`, the code divides by it, and neither is the
standard error of the mean that sits next to it.

#### What it costs

```
MMLU shape, 57 subjects of 100 samples. Two models compared at 95%:

  as reported today: StdErr 0.0490 -> needs a 13.58 point gap
  with the fix     : StdErr 0.0065 -> needs a  1.80 point gap

     gap  as reported today    with the fix
  ----------------------------------------------
     1pt  no difference        no difference
     2pt  no difference        significant           <- disagree
     3pt  no difference        significant           <- disagree
     5pt  no difference        significant           <- disagree
     8pt  no difference        significant           <- disagree
    13pt  no difference        significant           <- disagree
    14pt  significant          significant
```

Real model gaps live in the 2 to 8 point range. Every one of them is currently
reported as no difference.

### 2. Samples are counted twice

```python
for column in ["key", "subject"]:
    if len(data_subset[column].unique()) > 1:
        for name, _group in key_subject_mean.groupby([column]):
            ...
            std_err_mean_sum_of_squares += std**2 / num_samples
            std_err_mean_total_num_samples += num_samples
            std_err_mean_num_subjects += 1
```

The loop walks the same samples once per dimension and accumulates into the same
three variables, so when a metric has several keys **and** several subjects every
sample lands in the totals twice.

The upstream test suite already asserts the doubled value. From
`tests/tests_eval_framework/test_evaluation_generator.py`:

```python
"NumSamples metric1 - key1": 4,
"NumSamples metric1 - key2": 4,         # keys sum to 8
"NumSamples metric1 - subject1": 6,
"NumSamples metric1 - subject2": 2,     # subjects sum to 8
"NumSamples metric1": 16,               # 8 + 8
```

That fixture has 10 rows for `metric1`, 8 of them error free. Single-key metrics
in the same fixture come out right (`metric2` 3, `metric3` 2), which confirms the
doubling needs both dimensions.

This is live on **every multi-subject completion benchmark**, because
`FinishReason` emits three keys and `tasks/base.py` attaches it automatically to
every completion task.

### 3. One group of a single sample voids the whole figure

pandas `.std()` uses `ddof=1`, so a group holding one sample gives `NaN`. The
per-group write is guarded, but the accumulator one line below is not:

```python
if math.isnan(std) or num_samples == 0:
    aggregated_results[f"StdErr {metric} - {name[0]}"] = float("nan")
else:
    aggregated_results[f"StdErr {metric} - {name[0]}"] = std / np.sqrt(num_samples)
aggregated_results[f"NumSamples {metric} - {name[0]}"] = num_samples

std_err_mean_sum_of_squares += std**2 / num_samples   # NaN enters here
```

So:

```
57 subjects of 100 samples:           current 0.048793   fixed 0.006463
one subject reduced to a single sample: current nan        fixed 0.010692
                                        score is still fine: 0.599649
```

5601 good samples produce no usable standard error at all.

## Why all three survived

The only test of the combined figure asserts `NaN` for every metric in the
fixture:

```python
"StdErr metric1": float("nan"),
"StdErr metric2": float("nan"),
"StdErr metric3": float("nan"),
```

The three are a chain, not three separate bugs. Defect 3 is *why* the fixture
produces `NaN`; because the test asserts `NaN`, defect 1 was never compared
against a real number; and defect 2 corrupts the denominator defect 1 divides by.
Fixing defect 1 alone leaves the figure `NaN` in practice, and fixing defect 3
alone starts surfacing numbers that are 7.5x wrong. They have to go together.

## The fix

`patch/evaluation_generator.patch` applies to v0.15.2:

```bash
git apply patch/evaluation_generator.patch
```

It deletes the three accumulators and both branches that consumed them, and adds
one helper used in the two places that need it:

```python
def _standard_error_of_cell_mean(values: pd.DataFrame) -> tuple[float, int]:
    cells = values.groupby(["key", "subject"])["value"].agg(["std", "count"])
    cells = cells[cells["count"] > 0]
    num_samples = int(cells["count"].sum())

    measured = cells[(cells["count"] > 1) & cells["std"].notna()]
    if cells.empty or measured.empty:
        return float("nan"), num_samples

    degrees_of_freedom = float((measured["count"] - 1).sum())
    pooled_variance = float(((measured["count"] - 1) * measured["std"] ** 2).sum() / degrees_of_freedom)

    cell_variance = cells["std"].pow(2).where(cells["count"] > 1, pooled_variance).fillna(pooled_variance)
    variance_of_mean = float((cell_variance / cells["count"]).sum())

    return float(np.sqrt(variance_of_mean) / len(cells)), num_samples
```

Net 49 insertions, 42 deletions, and most of the additions are the docstring.
`ruff check` and `ruff format --check` both pass under the repo config.

Three decisions worth flagging:

- **Computed from the `(key, subject)` cells, not accumulated in the breakdown
  loop.** The cells are the groups the reported mean is actually taken over, and
  computing there is what removes the double counting rather than patching around
  it.
- **A single-sample cell borrows the pooled variance of the cells that have one.**
  Dropping such a cell would let it pass as zero uncertainty, which errs toward
  false confidence. Letting its `NaN` through is defect 3. Borrowing errs in
  neither direction. `NaN` is now returned only when no cell has two or more
  values, which is the one case where nothing can be estimated.
- **`NumSamples` now counts values rather than rows.** A standard error cannot
  divide by a count that includes rows holding no value. The upstream test
  already describes the micro average as "the plain mean of the 7 error-free
  values" while `NumSamples` counted 8 of them, so this lines the two up.

The per-group standard errors are fixed by the same helper. They had the same
shape of error whenever a breakdown group spanned more than one cell.

### No score moves

```
ok  upstream fixture          30 score keys identical
ok  57 subjects              117 score keys identical
ok  1 subject                  3 score keys identical
ok  one singleton subject     21 score keys identical
```

Every `Average`, `Average (micro)`, `Average (including Errors)` and
`ErrorFreeRatio` value is byte-identical before and after. Only `StdErr` and
`NumSamples` change.

### New values for the upstream test

`test_aggregate_results` will need these. Worked by hand in
`UPDATED_TEST_VALUES.md` so the numbers can be checked without running anything.

| key | current | fixed |
| --- | --- | --- |
| `StdErr metric1` | `nan` | `0.590726953281576` |
| `StdErr metric1 - key1` | `0.8660254037844386` | `0.9999999999999999` |
| `StdErr metric1 - key2` | `0.3535533905932738` | `0.4330127018922194` |
| `StdErr metric1 - subject1` | `0.4978909578906802` | `0.5590169943749475` |
| `StdErr metric2` | `nan` | `0` |
| `NumSamples metric1` | `16` | `7` |
| `NumSamples metric1 - key2` | `4` | `3` |
| `NumSamples metric1 - subject1` | `6` | `5` |
| `NumSamples metric2` | `3` | `2` |
| `NumSamples metric2 - subject2` | `1` | `0` |

Unchanged: `StdErr metric1 - subject2`, `StdErr metric3*`,
`NumSamples metric1 - key1`, `NumSamples metric1 - subject2`,
`NumSamples metric2 - subject1`, `NumSamples metric3*`.

## What is in here

| File | What it is |
| --- | --- |
| `aggregate_current.py` | `_aggregate_results` extracted verbatim from v0.15.2, byte for byte after removing one indent level |
| `aggregate_fixed.py` | the same function with the fix |
| `repro.py` | all three defects with the evidence, plus the proof that no score moves |
| `test_stderr.py` | the defects as tests: 12 pass on the fix, 9 fail on the shipped code |
| `patch/evaluation_generator.patch` | the fix against v0.15.2, `git apply`-ready |
| `UPDATED_TEST_VALUES.md` | the new upstream test numbers, worked by hand |

### On the extraction being trustworthy

The whole argument rests on `aggregate_current.py` being the real thing. Two
checks:

1. Its function body is **byte-for-byte identical** to upstream lines 141 to 274
   once one indent level is removed (the method became a free function so it can
   run without a task registry, an LLM and an output directory).
2. `repro.py` section 0 replays the upstream fixture through it and reproduces
   **every number the upstream test asserts**, including `NumSamples metric1 = 16`
   and `StdErr metric1 = nan`.

The patched upstream file was also compared back against `aggregate_fixed.py`:
the method bodies are identical, and the two helpers have identical ASTs once the
docstring and the private-name underscore are normalised. So the patch is exactly
the code the numbers above were measured on.

`mypy` is configured upstream but was not run here, since that needs the full
project environment.

## Attribution

`aggregate_current.py` and parts of `aggregate_fixed.py` are derived from
eval-framework, which is Apache-2.0. That license is included in `LICENSE` and
applies to the derived files here. Everything else in this repository is offered
under the same terms.

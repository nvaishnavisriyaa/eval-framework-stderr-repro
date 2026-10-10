# New values for `test_aggregate_results`

The fix changes `StdErr` and `NumSamples` in
`tests/tests_eval_framework/test_evaluation_generator.py::test_aggregate_results`.
Every `Average`, `Average (micro)`, `Average (including Errors)` and
`ErrorFreeRatio` assertion in that test stays exactly as it is.

Each number below is derived by hand further down, so it can be checked without
running anything. `gen_values.py` produced this file, so the arithmetic in the
tables and the derivations come from the same computation.

## Full comparison

| assertion | current | fixed | changes |
| --- | --- | --- | --- |
| `NumSamples metric1` | `16` | `7` | yes |
| `NumSamples metric1 - key1` | `4` | `4` | no |
| `NumSamples metric1 - key2` | `4` | `3` | yes |
| `NumSamples metric1 - subject1` | `6` | `5` | yes |
| `NumSamples metric1 - subject2` | `2` | `2` | no |
| `NumSamples metric2` | `3` | `2` | yes |
| `NumSamples metric2 - subject1` | `2` | `2` | no |
| `NumSamples metric2 - subject2` | `1` | `0` | yes |
| `NumSamples metric3` | `2` | `2` | no |
| `NumSamples metric3 - subject1` | `1` | `1` | no |
| `NumSamples metric3 - subject2` | `1` | `1` | no |
| `StdErr metric1` | `nan` | `0.590726953281576` | yes |
| `StdErr metric1 - key1` | `0.8660254037844386` | `0.9999999999999999` | yes |
| `StdErr metric1 - key2` | `0.3535533905932738` | `0.4330127018922194` | yes |
| `StdErr metric1 - subject1` | `0.4978909578906802` | `0.5590169943749475` | yes |
| `StdErr metric1 - subject2` | `nan` | `nan` | no |
| `StdErr metric2` | `nan` | `0` | yes |
| `StdErr metric2 - subject1` | `0` | `0` | no |
| `StdErr metric2 - subject2` | `nan` | `nan` | no |
| `StdErr metric3` | `nan` | `nan` | no |
| `StdErr metric3 - subject1` | `nan` | `nan` | no |
| `StdErr metric3 - subject2` | `nan` | `nan` | no |

## Where the metric1 numbers come from

### `StdErr metric1` and `NumSamples metric1`

| cell | n | values | variance |
| --- | --- | --- | --- |
| `key1` / `subject1` | 3 | [5.0, 2.0, 2.0] | 3 |
| `key1` / `subject2` | 1 | [4.0] | n/a, single sample |
| `key2` / `subject1` | 2 | [0.5, 1.5] | 0.5 |
| `key2` / `subject2` | 1 | [0.0] | n/a, single sample |

Pooled variance over the 2 cell(s) with n > 1, dof = 3:

    (2x3 + 1x0.5) / 3 = 2.166666667

Variance of each cell mean:

     key1 / subject1           3 / 3 = 1
     key1 / subject2     2.16667 / 1 = 2.166666667   (borrowed pooled)
     key2 / subject1         0.5 / 2 = 0.25
     key2 / subject2     2.16667 / 1 = 2.166666667   (borrowed pooled)

    StdErr = sqrt(5.583333333) / 4 cells = 0.590726953281576

Matches the implementation: `0.590726953281576`.

`NumSamples metric1` is the sum of the cell counts, 7, against `16` today.

### `StdErr metric1 - key1`

| cell | n | values | variance |
| --- | --- | --- | --- |
| `key1` / `subject1` | 3 | [5.0, 2.0, 2.0] | 3 |
| `key1` / `subject2` | 1 | [4.0] | n/a, single sample |

Pooled variance over the 1 cell(s) with n > 1, dof = 2:

    (2x3) / 2 = 3

Variance of each cell mean:

     key1 / subject1           3 / 3 = 1
     key1 / subject2           3 / 1 = 3   (borrowed pooled)

    StdErr = sqrt(4) / 2 cells = 1

Matches the implementation: `0.9999999999999999`.

### `StdErr metric1 - key2`

| cell | n | values | variance |
| --- | --- | --- | --- |
| `key2` / `subject1` | 2 | [0.5, 1.5] | 0.5 |
| `key2` / `subject2` | 1 | [0.0] | n/a, single sample |

Pooled variance over the 1 cell(s) with n > 1, dof = 1:

    (1x0.5) / 1 = 0.5

Variance of each cell mean:

     key2 / subject1         0.5 / 2 = 0.25
     key2 / subject2         0.5 / 1 = 0.5   (borrowed pooled)

    StdErr = sqrt(0.75) / 2 cells = 0.4330127018922193

Matches the implementation: `0.4330127018922194`.

### `StdErr metric1 - subject1`

| cell | n | values | variance |
| --- | --- | --- | --- |
| `key1` / `subject1` | 3 | [5.0, 2.0, 2.0] | 3 |
| `key2` / `subject1` | 2 | [0.5, 1.5] | 0.5 |

Pooled variance over the 2 cell(s) with n > 1, dof = 3:

    (2x3 + 1x0.5) / 3 = 2.166666667

Variance of each cell mean:

     key1 / subject1           3 / 3 = 1
     key2 / subject1         0.5 / 2 = 0.25

    StdErr = sqrt(1.25) / 2 cells = 0.5590169943749475

Matches the implementation: `0.5590169943749475`.

### `StdErr metric1 - subject2`

| cell | n | values | variance |
| --- | --- | --- | --- |
| `key1` / `subject2` | 1 | [4.0] | n/a, single sample |
| `key2` / `subject2` | 1 | [0.0] | n/a, single sample |

No cell holds two or more values, so nothing can be estimated and the
standard error is `nan`.

Matches the implementation: `nan`.

## A note on the two zero entries

`StdErr metric2` becomes `0` rather than `nan`. Its one cell with values holds
`[3.0, 3.0]`, a genuine variance of zero, so zero is the honest answer. The
other `metric2` cell holds a single row whose value is `None`, which contributes
nothing to the mean and so contributes nothing here either, which is why
`NumSamples metric2 - subject2` becomes `0`.

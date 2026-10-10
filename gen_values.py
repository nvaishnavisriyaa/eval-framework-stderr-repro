"""Generate UPDATED_TEST_VALUES.md, deriving every number rather than typing it."""

import math

import numpy as np

from aggregate_current import aggregate_results as current
from aggregate_fixed import aggregate_results as fixed
from repro import fixture_results

# metric1 of the upstream fixture, error-free rows only, grouped into (key, subject) cells.
# The third key1/subject1 row carries an Error and drops out. The key2/subject1 row whose
# value is None is error free but holds no value, so the pandas count skips it.
CELLS = {
    ("key1", "subject1"): [5.0, 2.0, 2.0],
    ("key1", "subject2"): [4.0],
    ("key2", "subject1"): [0.5, 1.5],
    ("key2", "subject2"): [0.0],
}


def variance(values):
    return float(np.var(values, ddof=1)) if len(values) > 1 else None


def derive(cells, label):
    """Work the helper by hand over ``cells`` and return the markdown plus the result."""
    out = [f"### {label}", ""]
    out.append("| cell | n | values | variance |")
    out.append("| --- | --- | --- | --- |")
    for key, values in cells.items():
        var = variance(values)
        shown = "n/a, single sample" if var is None else f"{var:g}"
        out.append(f"| `{key[0]}` / `{key[1]}` | {len(values)} | {values} | {shown} |")
    out.append("")

    measured = {k: v for k, v in cells.items() if len(v) > 1}
    if not measured:
        out.append("No cell holds two or more values, so nothing can be estimated and the")
        out.append("standard error is `nan`.")
        out.append("")
        return "\n".join(out), float("nan"), sum(len(v) for v in cells.values())

    dof = sum(len(v) - 1 for v in measured.values())
    pooled = sum((len(v) - 1) * variance(v) for v in measured.values()) / dof
    terms = " + ".join(f"{len(v) - 1}x{variance(v):g}" for v in measured.values())
    out.append(f"Pooled variance over the {len(measured)} cell(s) with n > 1, dof = {dof}:")
    out.append("")
    out.append(f"    ({terms}) / {dof} = {pooled:.10g}")
    out.append("")
    out.append("Variance of each cell mean:")
    out.append("")
    total = 0.0
    for key, values in cells.items():
        var = variance(values)
        used = pooled if var is None else var
        total += used / len(values)
        tag = "   (borrowed pooled)" if var is None else ""
        out.append(f"    {key[0]:>5} / {key[1]:<9} {used:>10.6g} / {len(values)} = {used / len(values):.10g}{tag}")
    result = math.sqrt(total) / len(cells)
    out.append("")
    out.append(f"    StdErr = sqrt({total:.10g}) / {len(cells)} cells = {result:.16g}")
    out.append("")
    return "\n".join(out), result, sum(len(v) for v in cells.values())


def main():
    cur, fix = current(fixture_results()), fixed(fixture_results())

    def show(value):
        if isinstance(value, float) and math.isnan(value):
            return "`nan`"
        if isinstance(value, float) and value == int(value) and abs(value) < 1e15:
            return f"`{int(value)}`" if float(value).is_integer() and value == 0 else f"`{value:.16g}`"
        return f"`{value:.16g}`" if isinstance(value, float) else f"`{value}`"

    lines = [
        "# New values for `test_aggregate_results`",
        "",
        "The fix changes `StdErr` and `NumSamples` in",
        "`tests/tests_eval_framework/test_evaluation_generator.py::test_aggregate_results`.",
        "Every `Average`, `Average (micro)`, `Average (including Errors)` and",
        "`ErrorFreeRatio` assertion in that test stays exactly as it is.",
        "",
        "Each number below is derived by hand further down, so it can be checked without",
        "running anything. `gen_values.py` produced this file, so the arithmetic in the",
        "tables and the derivations come from the same computation.",
        "",
        "## Full comparison",
        "",
        "| assertion | current | fixed | changes |",
        "| --- | --- | --- | --- |",
    ]
    keys = [k for k in sorted(set(cur) | set(fix)) if k.split()[0] in {"StdErr", "NumSamples"}]
    for key in keys:
        a, b = show(cur.get(key)), show(fix.get(key))
        lines.append(f"| `{key}` | {a} | {b} | {'yes' if a != b else 'no'} |")

    lines += ["", "## Where the metric1 numbers come from", ""]

    body, value, count = derive(CELLS, "`StdErr metric1` and `NumSamples metric1`")
    lines.append(body)
    lines.append(f"Matches the implementation: `{fix['StdErr metric1']:.16g}`.")
    lines.append("")
    lines.append(f"`NumSamples metric1` is the sum of the cell counts, {count}, against `16` today.")
    lines.append("")

    for name, subset in (
        ("`StdErr metric1 - key1`", {k: v for k, v in CELLS.items() if k[0] == "key1"}),
        ("`StdErr metric1 - key2`", {k: v for k, v in CELLS.items() if k[0] == "key2"}),
        ("`StdErr metric1 - subject1`", {k: v for k, v in CELLS.items() if k[1] == "subject1"}),
        ("`StdErr metric1 - subject2`", {k: v for k, v in CELLS.items() if k[1] == "subject2"}),
    ):
        body, value, count = derive(subset, name)
        lines.append(body)
        upstream_key = name.strip("`")
        actual = fix[upstream_key]
        shown = "nan" if math.isnan(actual) else f"{actual:.16g}"
        lines.append(f"Matches the implementation: `{shown}`.")
        lines.append("")

    lines += [
        "## A note on the two zero entries",
        "",
        "`StdErr metric2` becomes `0` rather than `nan`. Its one cell with values holds",
        "`[3.0, 3.0]`, a genuine variance of zero, so zero is the honest answer. The",
        "other `metric2` cell holds a single row whose value is `None`, which contributes",
        "nothing to the mean and so contributes nothing here either, which is why",
        "`NumSamples metric2 - subject2` becomes `0`.",
        "",
    ]

    with open("UPDATED_TEST_VALUES.md", "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    print("wrote UPDATED_TEST_VALUES.md")


if __name__ == "__main__":
    main()

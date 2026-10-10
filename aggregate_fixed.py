"""The same aggregation with the three standard-error defects fixed.

The only change is how the standard error and the sample count are produced.
Everything that sets ``Average``, ``Average (micro)``, ``Average (including
Errors)`` and ``ErrorFreeRatio`` is identical to ``aggregate_current.py``, so no
reported score moves.

The three accumulators (``std_err_mean_sum_of_squares``,
``std_err_mean_total_num_samples``, ``std_err_mean_num_subjects``) and both
branches that consumed them are gone, replaced by one helper called in the two
places that need it. That removes:

* the ``sqrt(sum / count)`` for ``sqrt(sum) / count`` mistake,
* the double counting from accumulating across both the key and the subject
  dimension,
* the dependence on ``std`` leaking out of the breakdown loop,

and it fixes the per-group standard errors, which had the same shape of error
whenever a breakdown group spanned more than one (key, subject) cell.

Net effect on the file: about 30 lines removed.
"""

import numpy as np
import pandas as pd

from result import Result


def standard_error_of_cell_mean(values: pd.DataFrame) -> tuple[float, int]:
    """Standard error of the unweighted mean of per (key, subject) cell means.

    ``Average {metric}`` is that unweighted mean, so the matching standard error
    is::

        StdErr = sqrt( sum_cells Var(cell mean) ) / number_of_cells
        Var(cell mean) = cell_variance / cell_n

    Returns the standard error and the number of values behind it.

    A cell holding a single sample has no variance of its own (pandas uses
    ddof=1) and borrows the pooled variance of the cells that do. Dropping such
    a cell would let it pass as zero uncertainty; letting its NaN through voided
    the whole figure. The standard error is NaN only when no cell holds two or
    more values, which is the one case where nothing can be estimated.

    ``values`` is expected to carry the ``key``, ``subject`` and ``value``
    columns, already filtered to the error-free rows.
    """
    cells = values.groupby(["key", "subject"])["value"].agg(["std", "count"])
    # A cell whose values are all None contributed nothing to the mean either.
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


def aggregate_results(results: list[Result]) -> dict[str, float | None]:
    data = pd.DataFrame(
        [
            {
                "metric_name": r.metric_name,
                "subject": r.subject,
                "key": r.key,
                "value": r.value,
                "error": r.error,
            }
            for r in results
        ]
    )
    if len(data) == 0:
        return {}
    data.fillna({"key": ""}, inplace=True)
    metrics = sorted(data["metric_name"].unique())
    aggregated_results: dict[str, float | None] = {}

    for metric in metrics:
        # filter for metric
        data_subset = data[data["metric_name"] == metric][["subject", "key", "value", "error"]]

        # filter and count errors
        total_count = len(data_subset)

        mask = data_subset["error"].isnull()
        data_subset_error_free = data_subset.loc[mask, ["subject", "key", "value"]]

        error_free_ratio = float(len(data_subset_error_free) / total_count)
        aggregated_results[f"ErrorFreeRatio {metric}"] = error_free_ratio

        # Default average is weighted by key and subject (macro average).
        # aggregate by key and subject first to have equal weights for all key / subject combinations
        key_subject_mean = data_subset_error_free.groupby(["key", "subject"]).mean()
        aggregated_results[f"Average {metric}"] = float(key_subject_mean[["value"]].mean()["value"])
        # Additionally, report the plain mean over all samples (micro average or per-sample average).
        aggregated_results[f"Average {metric} (micro)"] = float(data_subset_error_free["value"].mean())

        if error_free_ratio < 1.0:
            # Treat error samples (with value=None) as 0 for the "including errors" average
            data_subset_with_errors = data_subset[["key", "subject", "value", "error"]].copy()
            # Only fill value with 0 where there's an error (not for all None values)
            error_mask = data_subset_with_errors["error"].notna()
            data_subset_with_errors.loc[error_mask, "value"] = data_subset_with_errors.loc[
                error_mask, "value"
            ].fillna(0.0)
            key_subject_mean_with_errors = data_subset_with_errors.groupby(["key", "subject"])["value"].mean()
            aggregated_results[f"Average {metric} (including Errors)"] = float(key_subject_mean_with_errors.mean())

        for column in ["key", "subject"]:
            if len(data_subset[column].unique()) > 1:
                for name, _group in key_subject_mean.groupby([column]):
                    mask = data_subset[column] == name[0]
                    group = data_subset.loc[mask, ["subject", "key", "value", "error"]]
                    group_total_count = len(group)
                    group_error_free = group[group["error"].isnull()][["subject", "key", "value"]]
                    group_error_free_ratio = float(len(group_error_free) / group_total_count)
                    aggregated_results[f"ErrorFreeRatio {metric} - {name[0]}"] = group_error_free_ratio

                    group_key_subject_mean = group_error_free.groupby(["key", "subject"]).mean()
                    value = float(group_key_subject_mean[["value"]].mean()["value"])
                    aggregated_results[f"Average {metric} - {name[0]}"] = value

                    if group_error_free_ratio < 1.0:
                        # Treat error samples (with value=None) as 0 for the "including errors" average
                        group_with_errors = group[["key", "subject", "value", "error"]].copy()
                        # Only fill value with 0 where there's an error (not for all None values)
                        error_mask = group_with_errors["error"].notna()
                        group_with_errors.loc[error_mask, "value"] = group_with_errors.loc[
                            error_mask, "value"
                        ].fillna(0.0)
                        group_key_subject_mean_with_errors = group_with_errors.groupby(["key", "subject"])[
                            "value"
                        ].mean()
                        value_with_errors = float(group_key_subject_mean_with_errors.mean())
                        aggregated_results[f"Average {metric} (including Errors) - {name[0]}"] = value_with_errors

                    if not ("SequencePositions" in metric or "Bytes" in metric):
                        # The group reports the mean of its own cell means, so its standard
                        # error comes from the same helper as the combined figure.
                        group_std_err, group_num_samples = standard_error_of_cell_mean(group_error_free)
                        aggregated_results[f"StdErr {metric} - {name[0]}"] = group_std_err
                        aggregated_results[f"NumSamples {metric} - {name[0]}"] = group_num_samples

        if not ("SequencePositions" in metric or "Bytes" in metric):
            # Computed straight from the (key, subject) cells rather than accumulated
            # inside the breakdown loop above. The loop visits the same samples once per
            # dimension, so accumulating there counted every sample twice whenever both
            # dimensions had more than one value.
            std_err, num_samples = standard_error_of_cell_mean(data_subset_error_free)
            aggregated_results[f"StdErr {metric}"] = std_err
            aggregated_results[f"NumSamples {metric}"] = num_samples

    if (
        "Average Bytes" in aggregated_results
        and "Average SequencePositions" in aggregated_results
        and aggregated_results["Average Bytes"]
        and aggregated_results["Average SequencePositions"]
    ):
        aggregated_results["Average Bytes per Sequence Position"] = (
            aggregated_results["Average Bytes"] / aggregated_results["Average SequencePositions"]
        )

    return aggregated_results

"""Result aggregation from eval-framework, exactly as shipped.

Extracted verbatim from eval-framework v0.15.2 (commit 8c570ad),
``src/eval_framework/evaluation_generator.py``, method
``EvaluationGenerator._aggregate_results`` (lines 140 to 274).

Two things changed, neither of them logic:

1. The method became a free function, so it can run without constructing an
   ``EvaluationGenerator`` (which needs a task registry, an LLM and an output
   directory).
2. ``Result`` became the small local stand-in in ``result.py``, since only five
   of its fields are read here.

Every expression, branch and variable name inside the body is untouched.
``test_stderr.py`` checks that this copy still reproduces the numbers the
upstream test suite asserts, which is what makes it safe to reason about.
"""

import math

import numpy as np
import pandas as pd

from result import Result


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

        std_err_mean_sum_of_squares = 0.0
        std_err_mean_total_num_samples = 0.0
        std_err_mean_num_subjects = 0

        for column in ["key", "subject"]:
            if len(data_subset[column].unique()) > 1:
                for name, _group in key_subject_mean.groupby([column]):
                    mask = data_subset[column] == name[0]
                    group = data_subset.loc[mask, ["subject", "key", "value", "error"]]
                    # group = data_subset[data[column] == name][["subject", "key", "value", "error"]]
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
                        # calculate standard error for selected  metrics
                        group_key_subject_std = group_error_free.groupby(["key", "subject"]).std()
                        std = float(group_key_subject_std[["value"]].mean()["value"])
                        num_samples = len(group_error_free)

                        if math.isnan(std) or num_samples == 0:
                            aggregated_results[f"StdErr {metric} - {name[0]}"] = float("nan")
                        else:
                            aggregated_results[f"StdErr {metric} - {name[0]}"] = std / np.sqrt(num_samples)
                        aggregated_results[f"NumSamples {metric} - {name[0]}"] = num_samples

                        std_err_mean_sum_of_squares += std**2 / num_samples
                        std_err_mean_total_num_samples += num_samples
                        std_err_mean_num_subjects += 1

        if not ("SequencePositions" in metric or "Bytes" in metric):
            # calculate standard error for selected  metrics
            if std_err_mean_total_num_samples > 0:
                # calculate the standard error of the mean (SEM) for the aggregated results (eg. add in quadrature)
                # SEM = sqrt(sum(variance_i * n_i) / i)
                # where variance_i is the variance of each group and i is the number of groups
                # (the combined mean is also not weighted by the number of samples)
                if math.isnan(std) or std_err_mean_total_num_samples == 0:
                    aggregated_results[f"StdErr {metric}"] = float("nan")
                else:
                    aggregated_results[f"StdErr {metric}"] = np.sqrt(
                        std_err_mean_sum_of_squares / std_err_mean_num_subjects
                    )
                aggregated_results[f"NumSamples {metric}"] = std_err_mean_total_num_samples
            else:
                # if there are no sub-groups to combine, calculate the SEM here directly
                key_subject_std = data_subset_error_free.groupby(["key", "subject"]).std()
                std = float(key_subject_std[["value"]].mean()["value"])
                num_samples = len(data_subset_error_free)
                if math.isnan(std) or num_samples == 0:
                    aggregated_results[f"StdErr {metric}"] = float("nan")
                else:
                    aggregated_results[f"StdErr {metric}"] = std / np.sqrt(num_samples)
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

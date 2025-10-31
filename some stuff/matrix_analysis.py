# This file is here to provide the functions for matrix analysis
# on parameters which I found useful
import numpy as np
from hg_lambda_calc import get_lambda


def generate_interaction_matrix(n_motives=8, mean=0.0, sd=0.2):
    """Generate symmetric interaction matrix."""
    upper_triangle = np.random.normal(mean, sd, size=(n_motives, n_motives))
    inter_m = np.triu(upper_triangle, k=1)
    inter_m = inter_m + inter_m.T
    inter_m = np.clip(inter_m, -1, 1)
    np.fill_diagonal(inter_m, 0)

    return inter_m


def analyze_matrix_structure(inter_m):
    lambda_info = get_lambda(inter_m, decay_lambda=None)
    row_sums = np.sum(inter_m, axis=1)
    mean_row_sum = np.mean(row_sums)
    row_medians = np.median(inter_m, axis=1)
    row_variances = np.var(inter_m, axis=1)
    row_entropy = -np.nansum(
        (inter_m * np.log(np.abs(inter_m) + 1e-10)), axis=1
    )  # small constant to avoid log(0)
    n_positive = np.sum(inter_m > 0)  # maybe add SD here too
    n_negative = np.sum(inter_m < 0)
    n_neutral = np.sum(inter_m == 0)
    support_ratio = n_positive / (n_negative + n_positive)
    row_support_ratios = np.array(
        [
            np.sum(inter_m[i] > 0)
            / (np.sum(inter_m[i] < 0) + np.sum(inter_m[i] > 0) + 1e-10)
            for i in range(inter_m.shape[0])
        ]
    )
    conflict_ratio = n_negative / (n_negative + n_positive)
    row_conflict_ratios = np.array(
        [
            np.sum(inter_m[i] < 0)
            / (np.sum(inter_m[i] < 0) + np.sum(inter_m[i] > 0) + 1e-10)
            for i in range(inter_m.shape[0])
        ]
    )
    range_values = np.ptp(inter_m, axis=1)  # Peak to peak (max - min) for each row
    gini_coefficients = np.array(
        [
            (
                np.sum(np.abs(inter_m[i][:, None] - inter_m[i][None, :]))
                / (2 * len(inter_m[i]) * np.sum(np.abs(inter_m[i])) + 1e-10)
            )
            for i in range(inter_m.shape[0])
        ]
    )

    return {
        "lambda_info": lambda_info,
        "row_sums": row_sums,
        "mean_row_sum": mean_row_sum,
        "row_medians": row_medians,
        "row_variances": row_variances,
        "row_entropy": row_entropy,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_neutral": n_neutral,
        "support_ratio": support_ratio,
        "row_support_ratios": row_support_ratios,
        "conflict_ratio": conflict_ratio,
        "row_conflict_ratios": row_conflict_ratios,
        "range_values": range_values,
        "gini_coefficients": gini_coefficients,
    }

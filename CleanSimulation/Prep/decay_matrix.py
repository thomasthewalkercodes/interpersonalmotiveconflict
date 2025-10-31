# generating different decay stuff in a class
# later easy implementable into the interfaces tab
import numpy as np
import pandas as pd


class GenerateDecayMatrix:
    def matrix_specific_decay(inter_m, decay_lambda=None):
        n_motives = inter_m.shape[0]
        row_sums = np.sum(inter_m, axis=1)
        mean_row_sum = np.mean(row_sums)

        # Calculate equilibrium decay rate
        lambda_eq = (1 + mean_row_sum) / (n_motives - 1)

        # If no lambda provided, use equilibrium value
        if decay_lambda is None:
            decay_lambda = lambda_eq

        # Calculate stability ratio
        stability_ratio = (1 + mean_row_sum) / ((n_motives - 1) * decay_lambda)

        return {
            "decay_lambda": decay_lambda,
            "lambda_eq": lambda_eq,
            "stability_ratio": stability_ratio,
        }

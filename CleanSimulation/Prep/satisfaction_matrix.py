# This file is to create a class with different functions
# that later can be called into the interfaces tab to be used

import numpy as np
import pandas as pd


class SatisfactonMatrixGenerator:
    def normal_distribution_sat_matrix(n_motives=8, mean=0.2, sd=0.3):
        """Generate initial satisfaction matrix."""
        sat_values = np.random.normal(mean, sd, size=n_motives)
        sat_values = np.clip(sat_values, -1, 1)
        sat_values = np.round(sat_values, 3)
        sat_m = pd.DataFrame(
            [sat_values],
            columns=[f"motive_{i+1}" for i in range(n_motives)],
            index=["satisfaction"],
        )
        return sat_m

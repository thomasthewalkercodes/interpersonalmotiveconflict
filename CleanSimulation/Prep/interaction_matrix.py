# This file is to have a class of all interaction_matrix
# to then later be used in the interfaces

import numpy as np
import pandas as pd


class GenerateInteractionMatrix:
    def normal_distribution_int_matrix(n_motives=8, mean=0.0, sd=0.2):
        """Generate symmetric interaction matrix."""
        # Create a random interaction matrix
        matrix = np.random.normal(mean, sd, size=(n_motives, n_motives))
        matrix = (matrix + matrix.T) / 2  # Make it symmetric
        np.fill_diagonal(matrix, 0)  # No self-influence
        matrix = np.round(matrix, 3)  # Round to 3 decimal places for clarity
        return pd.DataFrame(
            matrix,
            columns=[f"motive_{i+1}" for i in range(n_motives)],
            index=[f"motive_{i+1}" for i in range(n_motives)],
        )

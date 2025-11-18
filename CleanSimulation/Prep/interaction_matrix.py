# This file is to have a class of all interaction_matrix
# to then later be used in the interfaces

import numpy as np
import pandas as pd


class GenerateInteractionMatrix:
    @staticmethod
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

    @staticmethod
    def circumplex_int_matrix(
        n_motives=8, start_motive=1, amplitude=0.3, elevation=0.1
    ):
        """Generate circumplex-based interaction matrix where the peak interaction is at the specified motive.

        Args:
            n_motives: Number of motives
            start_motive: Which motive (1-indexed) should have the peak interaction for each row
            amplitude: Range of interaction values
            elevation: Baseline interaction level
        """
        matrix = np.zeros((n_motives, n_motives))

        for i in range(n_motives):
            for j in range(n_motives):
                if i == j:
                    matrix[i, j] = 0  # No self-influence
                else:
                    # Calculate circular distance between motives
                    distance = min(abs(i - j), n_motives - abs(i - j))
                    # Apply angular displacement to rotate the peak to start_motive
                    angular_displacement = -(start_motive - 1) * (2 * np.pi / n_motives)
                    angle = distance * (2 * np.pi / n_motives) + angular_displacement
                    matrix[i, j] = amplitude * np.cos(angle) + elevation

        # Ensure symmetry
        matrix = (matrix + matrix.T) / 2
        # Round to 3 decimal places
        matrix = np.round(matrix, 3)

        return pd.DataFrame(
            matrix,
            columns=[f"motive_{i+1}" for i in range(n_motives)],
            index=[f"motive_{i+1}" for i in range(n_motives)],
        )


if __name__ == "__main__":
    generator = GenerateInteractionMatrix()

    print("Circumplex Interaction Matrix:")
    circumplex_matrix = generator.circumplex_int_matrix(
        n_motives=8, amplitude=0.2, elevation=0
    )
    print(circumplex_matrix)
    print(
        "\nNote: Adjacent motives (e.g., 1-2, 2-3, 8-1) have highest interaction values"
    )
    print("Opposite motives (e.g., 1-5, 2-6) have lowest interaction values")

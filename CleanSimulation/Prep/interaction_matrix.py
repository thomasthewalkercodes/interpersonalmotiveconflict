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
        n_motives=8, start_motive=1, amplitude=0.2, elevation=0.1
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

    @staticmethod
    def borderline_int_matrix(
        n_motives=8,
        start_motive=1,
        amplitude_dict=None,
        elevation_dict=None,
        base_amplitude=0.2,
        base_elevation=0.1,
    ):
        """Generate circumplex matrix with individual amplitude/elevation control for each motive.

        Creates a circumplex matrix where you can specify different amplitude and elevation
        for each motive individually. Motives not specified use the base values.

        Args:
            n_motives: Number of motives (default 8)
            start_motive: Which motive (1-indexed) should have the peak interaction (default 1)
            amplitude_dict: Dictionary mapping motive number (1-indexed) to amplitude
                           e.g., {3: 0.6, 7: 0.5} (default None)
            elevation_dict: Dictionary mapping motive number (1-indexed) to elevation
                           e.g., {3: -0.1, 7: 0.0} (default None)
            base_amplitude: Default amplitude for motives not in amplitude_dict (default 0.2)
            base_elevation: Default elevation for motives not in elevation_dict (default 0.1)
        """
        if amplitude_dict is None:
            amplitude_dict = {}
        if elevation_dict is None:
            elevation_dict = {}

        matrix = np.zeros((n_motives, n_motives))

        # Create circumplex matrix with individual amplitude/elevation per motive
        for i in range(n_motives):
            motive_num = i + 1  # Convert to 1-indexed

            # Get amplitude and elevation for this motive
            amplitude = amplitude_dict.get(motive_num, base_amplitude)
            elevation = elevation_dict.get(motive_num, base_elevation)

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

    print("\n" + "=" * 60)
    print("\nBorderline Interaction Matrix (custom amplitudes for motives 3 and 6):")
    borderline_matrix = generator.borderline_int_matrix(
        n_motives=8,
        start_motive=1,
        amplitude_dict={3: 0.2, 6: 0.2},  # Custom amplitudes for motives 3 and 6
        elevation_dict={3: 0.1, 6: 0.1},  # Custom elevations for motives 3 and 6
        base_amplitude=0.2,
        base_elevation=0.0,
    )
    print(borderline_matrix)
    print("\nNote: Each motive can have its own amplitude and elevation")
    print("Motive 3: amplitude=0.6, elevation=-0.1")
    print("Motive 6: amplitude=0.5, elevation=-0.05")
    print("Others: amplitude=0.2, elevation=0.0")
    print("Compare row 3 and row 6 - different amplitudes and baselines!")

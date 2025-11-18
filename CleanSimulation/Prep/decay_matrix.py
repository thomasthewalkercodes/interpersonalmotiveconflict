# generating different decay stuff in a class
# later easy implementable into the interfaces tab
import numpy as np
import pandas as pd


class GenerateDecayMatrix:

    @staticmethod
    def matrix_specific_decay(inter_m, decay_lambda=None):
        n_motives = inter_m.shape[0]
        row_sums = np.sum(inter_m, axis=1)
        mean_row_sum = np.mean(row_sums)

        # Calculate equilibrium decay rate
        lambda_eq = (1 + mean_row_sum) / (n_motives - 1)

        # If no lambda provided, use equilibrium value
        if decay_lambda is None:
            decay_lambda = lambda_eq

        decay_lambda = round(decay_lambda, 3)

        return {decay_lambda}

    @staticmethod
    def individual_decay_sin(n_motives=8, start_motive=1, amplitude=0.1, elevation=0.2):
        # Motive 1 starts at angle 0, motive 2 at 2π/n_motives, etc.
        angular_displacement = -(start_motive - 1) * (2 * np.pi / n_motives)

        cat_angles = np.linspace(
            angular_displacement,
            angular_displacement + 2 * np.pi,
            n_motives,
            endpoint=False,
        )

        # Calculate decay rates based on angles + round
        decay_values = np.round(amplitude * np.cos(cat_angles) + elevation, 3)
        return pd.DataFrame(
            [decay_values],
            columns=[f"motive_{i+1}" for i in range(n_motives)],
            index=["decay_rate"],
        )


if __name__ == "__main__":
    generator = GenerateDecayMatrix()  # Create an instance of the class
    decay_df = generator.individual_decay_sin(
        start_motive=3, amplitude=0.01, elevation=0.02
    )  # Call the method
    print("Decay Values Arranged in Circumplex Pattern:")
    print(decay_df)
# works

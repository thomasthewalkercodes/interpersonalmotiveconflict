import numpy as np


def generate_interaction_matrix(n_motives=8, mean=0.0, sd=0.2):
    """Generate symmetric interaction matrix."""
    upper_triangle = np.random.normal(mean, sd, size=(n_motives, n_motives))
    inter_m = np.triu(upper_triangle, k=1)
    inter_m = inter_m + inter_m.T
    inter_m = np.clip(inter_m, -1, 1)
    np.fill_diagonal(inter_m, 0)

    return inter_m


def get_lambda(inter_m, decay_lambda=None):
    """
    Analyze interaction matrix stability and return constant decay λ.
    get different ratios for different lambdas (if given or recommended one)

    Parameters:
    - inter_m: interaction matrix (symmetric, zero diagonal)
    - decay_lambda: if None, calculates recommended value

    Returns:
    - perfect lambda for the specific matrix
    - stability parameters (under 1 is saturated over 1 is bleeding)
    """
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


# example usage
if __name__ == "__main__":
    print("EXAMPLE USAGE OF INTERACTION MATRIX AND LAMBDA CALCULATION")
    inter_m = generate_interaction_matrix(n_motives=8, mean=0, sd=0.1)
    print("Generated Interaction Matrix:")
    print(inter_m.round(3))

    lambda_info = get_lambda(inter_m, decay_lambda=None)
    print("\nCalculated Lambda Information:")
    for key, value in lambda_info.items():
        print(f"{key}: {value:.4f}")

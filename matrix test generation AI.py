import numpy as np


def generate_motive_matrix(n_motives=8, mean=0.0, sd=0.2, seed=None):
    """
    Generate a symmetric congruence matrix for motives.

    Parameters:
    - n_motives: number of motives (default 8)
    - mean: mean of the normal distribution (default 0.0)
    - sd: standard deviation (default 0.2)
    - seed: random seed for reproducibility

    Returns:
    - M: symmetric matrix with zeros on diagonal, values clipped to [-1, 1]
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate upper triangle from normal distribution
    upper_triangle = np.random.normal(mean, sd, size=(n_motives, n_motives))

    # Make symmetric by mirroring upper triangle to lower
    M = np.triu(upper_triangle, k=1)  # Upper triangle excluding diagonal
    M = M + M.T  # Add transpose to make symmetric

    # Clip values to [-1, 1]
    M = np.clip(M, -1, 1)

    # Ensure diagonal is zero (motive doesn't affect itself)
    np.fill_diagonal(M, 0)

    return M


def analyze_matrix_stability(M, decay_coef=0.15):
    """
    Analyze a congruence matrix for stability/bleeding characteristics.

    Parameters:
    - M: congruence matrix (symmetric, zero diagonal)
    - decay_coef: decay coefficient (default 0.15)

    Returns:
    - Dictionary with analysis results
    """
    n_motives = M.shape[0]

    # Calculate row sums (excluding diagonal, which is 0 anyway)
    row_sums = np.sum(M, axis=1)
    mean_row_sum = np.mean(row_sums)

    # Calculate minimum decay coefficient to prevent bleeding
    decay_coef_min = (1 + mean_row_sum) / (2 * (n_motives - 1))

    # Calculate bleeding risk with current decay_coef
    bleeding_risk = (1 + mean_row_sum) / ((n_motives - 1) * decay_coef)

    # Calculate equilibrium mean satisfaction
    mean_sat_eq = 1 - (1 + mean_row_sum) / ((n_motives - 1) * decay_coef)

    # Determine system state
    if bleeding_risk > 2:
        state = "BLEEDING - system will crash"
    elif bleeding_risk > 1.5:
        state = "UNSTABLE - very low equilibrium"
    elif bleeding_risk > 1.0:
        state = "MARGINAL - low equilibrium"
    else:
        state = "STABLE - healthy equilibrium"

    return {
        "n_motives": n_motives,
        "mean_row_sum": mean_row_sum,
        "row_sums": row_sums,
        "decay_coef": decay_coef,
        "decay_coef_min": decay_coef_min,
        "bleeding_risk": bleeding_risk,
        "mean_sat_eq": mean_sat_eq,
        "state": state,
    }


def print_analysis(analysis):
    """Pretty print the analysis results."""
    print(f"\n{'='*60}")
    print(f"MATRIX STABILITY ANALYSIS")
    print(f"{'='*60}")
    print(f"Number of motives: {analysis['n_motives']}")
    print(f"Mean row sum: {analysis['mean_row_sum']:.4f}")
    print(f"\nRow sums per motive:")
    for i, rs in enumerate(analysis["row_sums"]):
        print(f"  Motive {i}: {rs:.4f}")
    print(f"\n{'─'*60}")
    print(f"Current decay coefficient: {analysis['decay_coef']:.4f}")
    print(f"Minimum decay coefficient: {analysis['decay_coef_min']:.4f}")
    print(f"\nBleeding risk: {analysis['bleeding_risk']:.4f}")
    print(f"  (>2.0 = bleeding, <2.0 = stable)")
    print(f"\nEquilibrium mean satisfaction: {analysis['mean_sat_eq']:.4f}")
    print(f"  (range: -1.0 to +1.0)")
    print(f"\nSystem state: {analysis['state']}")
    print(f"{'='*60}\n")


# Example usage
if __name__ == "__main__":
    # Generate a single matrix
    print("EXAMPLE 1: Single matrix analysis")
    M = generate_motive_matrix(n_motives=8, mean=-0.5, sd=0.2, seed=42)

    print("\nGenerated Matrix:")
    print(M)

    # Analyze with default decay coefficient
    analysis = analyze_matrix_stability(M, decay_coef=0.15)
    print_analysis(analysis)

    # Test with the minimum decay coefficient
    print("\nTesting with minimum decay coefficient:")
    analysis_min = analyze_matrix_stability(M, decay_coef=analysis["decay_coef_min"])
    print_analysis(analysis_min)

    # Generate multiple matrices and summarize
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Multiple matrix generation")
    print("=" * 60)

    n_samples = 20
    results = []

    for i in range(n_samples):
        M = generate_motive_matrix(n_motives=8, mean=0, sd=0.5)
        analysis = analyze_matrix_stability(M, decay_coef=0.15)
        results.append(analysis)

    print(f"\nGenerated {n_samples} matrices with mean=0.0, sd=0.2")
    print(f"\nSummary statistics:")
    print(
        f"  Mean row sum range: [{min(r['mean_row_sum'] for r in results):.4f}, "
        f"{max(r['mean_row_sum'] for r in results):.4f}]"
    )
    print(
        f"  Min decay_coef needed: [{min(r['decay_coef_min'] for r in results):.4f}, "
        f"{max(r['decay_coef_min'] for r in results):.4f}]"
    )
    print(
        f"  Equilibrium sat range: [{min(r['mean_sat_eq'] for r in results):.4f}, "
        f"{max(r['mean_sat_eq'] for r in results):.4f}]"
    )

    stable_count = sum(1 for r in results if r["bleeding_risk"] < 2.0)
    print(f"\nWith decay_coef=0.15: {stable_count}/{n_samples} matrices are stable")

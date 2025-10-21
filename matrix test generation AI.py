import numpy as np


def generate_motive_matrix(n_motives=8, mean=0.0, sd=0.2, seed=None):
    """Generate symmetric congruence matrix."""
    if seed is not None:
        np.random.seed(seed)

    upper_triangle = np.random.normal(mean, sd, size=(n_motives, n_motives))
    M = np.triu(upper_triangle, k=1)
    M = M + M.T
    M = np.clip(M, -1, 1)
    np.fill_diagonal(M, 0)

    return M


def analyze_matrix_stability(M, decay_lambda=None):
    """
    Analyze congruence matrix stability with constant decay λ.

    Parameters:
    - M: congruence matrix (symmetric, zero diagonal)
    - decay_lambda: if None, calculates recommended value

    Returns:
    - Dictionary with stability analysis
    """
    n_motives = M.shape[0]

    row_sums = np.sum(M, axis=1)
    mean_row_sum = np.mean(row_sums)

    # Calculate equilibrium decay rate
    lambda_eq = (1 + mean_row_sum) / (n_motives - 1)

    # Safe operating range
    lambda_min = 0.5 * lambda_eq
    lambda_max = 2.0 * lambda_eq

    # If no lambda provided, use equilibrium value
    if decay_lambda is None:
        decay_lambda = lambda_eq

    # Calculate stability ratio
    stability_ratio = (1 + mean_row_sum) / ((n_motives - 1) * decay_lambda)

    # Determine system state
    if stability_ratio > 2.0:
        state = "BLEEDING - system crashes to -1"
        health = "CRITICAL"
    elif stability_ratio > 1.5:
        state = "UNSTABLE - very low equilibrium"
        health = "POOR"
    elif stability_ratio > 1.2:
        state = "LOW EQUILIBRIUM - sustainable but low satisfaction"
        health = "FAIR"
    elif stability_ratio > 0.8:
        state = "HEALTHY EQUILIBRIUM - good balance"
        health = "GOOD"
    elif stability_ratio > 0.5:
        state = "HIGH EQUILIBRIUM - high satisfaction"
        health = "GOOD"
    else:
        state = "SATURATION RISK - may freeze at +1"
        health = "CAUTION"

    return {
        "n_motives": n_motives,
        "mean_row_sum": mean_row_sum,
        "row_sums": row_sums,
        "lambda_equilibrium": lambda_eq,
        "lambda_min_safe": lambda_min,
        "lambda_max_safe": lambda_max,
        "lambda_current": decay_lambda,
        "stability_ratio": stability_ratio,
        "state": state,
        "health": health,
        "is_in_safe_range": lambda_min <= decay_lambda <= lambda_max,
    }


def print_analysis(analysis):
    """Pretty print the analysis results."""
    print(f"\n{'='*70}")
    print(f"MATRIX STABILITY ANALYSIS")
    print(f"{'='*70}")
    print(f"Number of motives (N): {analysis['n_motives']}")
    print(f"Mean row sum of M: {analysis['mean_row_sum']:.4f}")
    print(f"\nRow sums per motive:")
    for i, rs in enumerate(analysis["row_sums"]):
        print(f"  Motive {i}: {rs:+.4f}")

    print(f"\n{'─'*70}")
    print(f"DECAY PARAMETER (λ) ANALYSIS")
    print(f"{'─'*70}")
    print(f"Equilibrium λ:     {analysis['lambda_equilibrium']:.4f} (perfect balance)")
    print(
        f"Safe range:        [{analysis['lambda_min_safe']:.4f}, {analysis['lambda_max_safe']:.4f}]"
    )
    print(f"Current λ:         {analysis['lambda_current']:.4f}")
    print(f"In safe range?     {'✓ YES' if analysis['is_in_safe_range'] else '✗ NO'}")

    print(f"\n{'─'*70}")
    print(f"STABILITY ASSESSMENT")
    print(f"{'─'*70}")
    print(f"Stability ratio (R): {analysis['stability_ratio']:.4f}")
    print(f"  R > 2.0  → Bleeding")
    print(f"  R ≈ 1.0  → Equilibrium")
    print(f"  R < 0.5  → Saturation")
    print(f"\nSystem state: {analysis['state']}")
    print(f"Health: {analysis['health']}")
    print(f"{'='*70}\n")


def compare_lambda_values(M, lambda_values):
    """Compare stability across different λ values."""
    print(f"\n{'='*70}")
    print(f"COMPARING DIFFERENT λ VALUES")
    print(f"{'='*70}\n")

    results = []
    for lam in lambda_values:
        analysis = analyze_matrix_stability(M, decay_lambda=lam)
        results.append(analysis)
        print(
            f"λ = {lam:.3f} → R = {analysis['stability_ratio']:.3f} → {analysis['health']}"
        )

    return results


# Example usage
if __name__ == "__main__":
    print("EXAMPLE 1: Single matrix with recommended λ")
    M = generate_motive_matrix(n_motives=8, mean=0.0, sd=0.2, seed=1)

    print("\nGenerated Matrix M:")
    print(M.round(3))

    # Analyze with recommended λ
    analysis = analyze_matrix_stability(M)
    print_analysis(analysis)

    # Compare different λ values
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Testing different λ values")
    lambda_test = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    compare_lambda_values(M, lambda_test)

    # Generate multiple matrices and find λ ranges
    print("\n" + "=" * 70)
    print("EXAMPLE 3: λ ranges across 20 random matrices")
    print("=" * 70)

    n_samples = 20
    lambda_eqs = []
    lambda_mins = []
    lambda_maxs = []

    for i in range(n_samples):
        M_sample = generate_motive_matrix(n_motives=8, mean=0.0, sd=0.2)
        analysis = analyze_matrix_stability(M_sample)
        lambda_eqs.append(analysis["lambda_equilibrium"])
        lambda_mins.append(analysis["lambda_min_safe"])
        lambda_maxs.append(analysis["lambda_max_safe"])

    print(f"\nFor matrices with mean=0.0, sd=0.2, N=8:")
    print(f"  λ_equilibrium range: [{min(lambda_eqs):.4f}, {max(lambda_eqs):.4f}]")
    print(f"  λ_min_safe range:    [{min(lambda_mins):.4f}, {max(lambda_mins):.4f}]")
    print(f"  λ_max_safe range:    [{min(lambda_maxs):.4f}, {max(lambda_maxs):.4f}]")
    print(
        f"\nRecommended λ for this distribution: {np.mean(lambda_eqs):.4f} ± {np.std(lambda_eqs):.4f}"
    )

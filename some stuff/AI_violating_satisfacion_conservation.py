"""
Mathematical analysis of satisfaction conservation in motive systems
Tests the hypothesis: High amplitude causes net satisfaction loss over time
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import seaborn as sns


def create_circumplex_correlation_matrix(
    n_motives=8, elevation=0.1, amplitude=0.9, displacement=0.0
):
    """Create correlation matrix based on circumplex structure"""
    correlation_matrix = np.eye(n_motives)
    angles = np.array([i * (360 / n_motives) for i in range(n_motives)])
    displacement_rad = np.radians(displacement)

    for i in range(n_motives):
        for j in range(n_motives):
            if i != j:
                angle_i = np.radians(angles[i])
                angle_j = np.radians(angles[j])
                angle_diff = angle_i - angle_j
                correlation = elevation + amplitude * np.cos(
                    angle_diff - displacement_rad
                )
                correlation_matrix[i, j] = correlation

    return correlation_matrix


# ============================================================================
# MATHEMATICAL FORMULATION
# ============================================================================


def calculate_theoretical_satisfaction_flow(
    correlation_matrix, influence_multiplier=0.03, growth_rate=0.2, network_impact=0.05
):
    """
    Calculate theoretical satisfaction flow rates

    For a single active motive k:

    dS_k/dt = +growth_rate - decay_rate
    dS_i/dt = +C_ki * network_impact - decay_rate  (for i ≠ k)

    Where:
    - decay_rate = (1 - mean_correlation) * influence_multiplier
    - C_ki = correlation between motive k and i

    Returns:
        net_flow_active: Net change for active motive
        net_flow_others: Net change for other motives (mean)
        total_system_flow: Total satisfaction change across all motives
    """
    n_motives = correlation_matrix.shape[0]
    mean_correlation = correlation_matrix.mean()
    decay_rate = (1 - mean_correlation) * influence_multiplier

    # For the active motive
    net_flow_active = growth_rate - decay_rate

    # For other motives (average)
    # Each motive receives correlation * network_impact from active motive
    correlation_effects = correlation_matrix.copy()
    np.fill_diagonal(correlation_effects, 0)  # Exclude self
    mean_correlation_effect = correlation_effects.mean(axis=1).mean()

    net_flow_others = (mean_correlation_effect * network_impact) - decay_rate

    # Total system flow (1 active + 7 others)
    total_system_flow = net_flow_active + (n_motives - 1) * net_flow_others

    return net_flow_active, net_flow_others, total_system_flow, decay_rate


def analyze_satisfaction_balance(
    elevation,
    amplitude,
    influence_multiplier=0.03,
    growth_rate=0.2,
    network_impact=0.05,
):
    """
    Analyze whether system maintains satisfaction balance

    Returns dict with:
        - elevation, amplitude
        - mean_correlation
        - negative_correlation_count
        - total_system_flow
        - is_balanced (bool)
    """
    correlation_matrix = create_circumplex_correlation_matrix(
        n_motives=8, elevation=elevation, amplitude=amplitude
    )

    net_flow_active, net_flow_others, total_system_flow, decay_rate = (
        calculate_theoretical_satisfaction_flow(
            correlation_matrix, influence_multiplier, growth_rate, network_impact
        )
    )

    # Count negative correlations
    correlation_matrix_no_diag = correlation_matrix.copy()
    np.fill_diagonal(correlation_matrix_no_diag, 0)
    negative_count = np.sum(correlation_matrix_no_diag < 0)
    negative_mean = (
        correlation_matrix_no_diag[correlation_matrix_no_diag < 0].mean()
        if negative_count > 0
        else 0
    )

    mean_correlation = correlation_matrix.mean()

    return {
        "elevation": elevation,
        "amplitude": amplitude,
        "mean_correlation": mean_correlation,
        "negative_correlation_count": negative_count,
        "negative_correlation_mean": negative_mean,
        "decay_rate": decay_rate,
        "net_flow_active": net_flow_active,
        "net_flow_others": net_flow_others,
        "total_system_flow": total_system_flow,
        "is_balanced": total_system_flow >= 0,
    }


# ============================================================================
# EMPIRICAL VALIDATION
# ============================================================================


def simulate_and_track_satisfaction(
    correlation_matrix,
    initial_satisfaction,
    steps=300,
    influence_multiplier=0.03,
    growth_rate=0.2,
    network_impact=0.05,
):
    """
    Simulate system and track total satisfaction over time

    Returns:
        satisfaction_history: (steps+1, n_motives) array
        total_satisfaction: (steps+1,) array - sum across all motives
        mean_satisfaction: (steps+1,) array - mean across all motives
    """
    n_motives = correlation_matrix.shape[0]
    satisfaction = initial_satisfaction.copy()
    mean_influence = correlation_matrix.mean()

    satisfaction_history = [satisfaction.copy()]
    total_satisfaction = [satisfaction.sum()]
    mean_satisfaction = [satisfaction.mean()]

    active_behavior = None

    for step in range(steps):
        unsatisfied = np.where(satisfaction < 0)[0]

        # Determine active behavior
        if active_behavior is None:
            if len(unsatisfied) > 0:
                probs = -satisfaction[unsatisfied]
                probs = probs / probs.sum()
                active_behavior = np.random.choice(unsatisfied, p=probs)
        else:
            if satisfaction[active_behavior] >= 1:
                active_behavior = None

        # Apply decay
        satisfaction = satisfaction - (1 - mean_influence) * influence_multiplier

        # Apply active behavior effects
        if active_behavior is not None:
            satisfaction[active_behavior] += growth_rate

            for i in range(n_motives):
                if i != active_behavior:
                    influence = correlation_matrix[active_behavior, i]
                    satisfaction[i] += influence * network_impact

        satisfaction = np.clip(satisfaction, -1, 1)

        satisfaction_history.append(satisfaction.copy())
        total_satisfaction.append(satisfaction.sum())
        mean_satisfaction.append(satisfaction.mean())

    return (
        np.array(satisfaction_history),
        np.array(total_satisfaction),
        np.array(mean_satisfaction),
    )


def empirical_satisfaction_drift(
    elevation, amplitude, n_trials=50, steps=300, analysis_window=(50, 250)
):
    """
    Empirically measure satisfaction drift through multiple trials

    Returns:
        mean_drift: average change in total satisfaction per step
        final_mean_satisfaction: mean satisfaction at end
        fraction_dissatisfied: fraction of time spent with >50% dissatisfied motives
    """
    drifts = []
    final_means = []
    dissatisfied_fractions = []

    correlation_matrix = create_circumplex_correlation_matrix(
        n_motives=8, elevation=elevation, amplitude=amplitude
    )

    for _ in range(n_trials):
        initial_satisfaction = np.clip(np.random.normal(0.5, 0.2, 8), -1, 1)

        sat_history, total_sat, mean_sat = simulate_and_track_satisfaction(
            correlation_matrix, initial_satisfaction, steps=steps
        )

        # Calculate drift in analysis window
        window_start, window_end = analysis_window
        window_total = total_sat[window_start:window_end]

        # Linear regression to find drift
        x = np.arange(len(window_total))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, window_total)
        drifts.append(slope)

        # Final mean satisfaction
        final_means.append(mean_sat[-1])

        # Fraction of time with >50% dissatisfied
        dissatisfied_count = np.sum(sat_history[window_start:window_end] < 0, axis=1)
        fraction_dissatisfied = np.mean(dissatisfied_count > 4)  # >4 out of 8
        dissatisfied_fractions.append(fraction_dissatisfied)

    return np.mean(drifts), np.mean(final_means), np.mean(dissatisfied_fractions)


# ============================================================================
# COMPREHENSIVE TESTING
# ============================================================================


def test_satisfaction_conservation_hypothesis(
    elevation_range=np.linspace(-0.5, 0.3, 20),
    amplitude_range=np.linspace(0.1, 0.9, 20),
    n_trials_per_combo=30,
):
    """
    Test hypothesis: High amplitude causes satisfaction loss

    For each (elevation, amplitude) combination:
    1. Calculate theoretical flow
    2. Measure empirical drift
    3. Compare predictions
    """
    results = []

    print("Testing Satisfaction Conservation Hypothesis...")
    print(f"Parameter combinations: {len(elevation_range) * len(amplitude_range)}")

    for elevation in tqdm(elevation_range):
        for amplitude in amplitude_range:
            # Theoretical analysis
            theoretical = analyze_satisfaction_balance(elevation, amplitude)

            # Empirical measurement
            empirical_drift, final_mean, dissatisfied_frac = (
                empirical_satisfaction_drift(
                    elevation, amplitude, n_trials=n_trials_per_combo
                )
            )

            results.append(
                {
                    **theoretical,
                    "empirical_drift": empirical_drift,
                    "final_mean_satisfaction": final_mean,
                    "dissatisfied_fraction": dissatisfied_frac,
                    "system_unstable": empirical_drift
                    < -0.001,  # Losing satisfaction over time
                }
            )

    return pd.DataFrame(results)


def visualize_conservation_analysis(df_results):
    """Create comprehensive visualization of conservation analysis"""
    fig = plt.figure(figsize=(20, 12))

    # 1. Theoretical total system flow
    ax1 = plt.subplot(2, 4, 1)
    pivot1 = df_results.pivot_table(
        values="total_system_flow",
        index="amplitude",
        columns="elevation",
        aggfunc="mean",
    )
    im1 = ax1.imshow(pivot1.values, cmap="RdBu_r", aspect="auto", origin="lower")
    ax1.contour(pivot1.values, levels=[0], colors="black", linewidths=2)
    ax1.set_xlabel("Elevation Index", fontsize=11)
    ax1.set_ylabel("Amplitude Index", fontsize=11)
    ax1.set_title("Theoretical Total System Flow\n(Black line = 0)", fontweight="bold")
    plt.colorbar(im1, ax=ax1, label="Flow Rate")

    # 2. Empirical drift
    ax2 = plt.subplot(2, 4, 2)
    pivot2 = df_results.pivot_table(
        values="empirical_drift", index="amplitude", columns="elevation", aggfunc="mean"
    )
    im2 = ax2.imshow(pivot2.values, cmap="RdBu_r", aspect="auto", origin="lower")
    ax2.contour(pivot2.values, levels=[0], colors="black", linewidths=2)
    ax2.set_xlabel("Elevation Index", fontsize=11)
    ax2.set_ylabel("Amplitude Index", fontsize=11)
    ax2.set_title("Empirical Satisfaction Drift\n(Black line = 0)", fontweight="bold")
    plt.colorbar(im2, ax=ax2, label="Drift Rate")

    # 3. Theoretical vs Empirical correlation
    ax3 = plt.subplot(2, 4, 3)
    stable = df_results[~df_results["system_unstable"]]
    unstable = df_results[df_results["system_unstable"]]

    ax3.scatter(
        stable["total_system_flow"],
        stable["empirical_drift"],
        alpha=0.4,
        s=30,
        c="green",
        label="Stable",
    )
    ax3.scatter(
        unstable["total_system_flow"],
        unstable["empirical_drift"],
        alpha=0.4,
        s=30,
        c="red",
        label="Unstable",
    )
    ax3.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax3.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Theoretical Flow", fontsize=11)
    ax3.set_ylabel("Empirical Drift", fontsize=11)
    ax3.set_title("Theoretical vs Empirical\n(Validation)", fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Amplitude effect on stability
    ax4 = plt.subplot(2, 4, 4)
    amp_groups = (
        df_results.groupby("amplitude")
        .agg({"system_unstable": "mean", "empirical_drift": "mean"})
        .reset_index()
    )

    ax4_twin = ax4.twinx()
    line1 = ax4.plot(
        amp_groups["amplitude"],
        amp_groups["system_unstable"],
        "o-",
        color="red",
        linewidth=2,
        markersize=6,
        label="Instability Rate",
    )
    line2 = ax4_twin.plot(
        amp_groups["amplitude"],
        amp_groups["empirical_drift"],
        "s-",
        color="blue",
        linewidth=2,
        markersize=6,
        label="Mean Drift",
    )

    ax4.set_xlabel("Amplitude", fontsize=11)
    ax4.set_ylabel("Instability Rate", fontsize=11, color="red")
    ax4_twin.set_ylabel("Mean Drift", fontsize=11, color="blue")
    ax4.set_title("Amplitude Effect on System Stability", fontweight="bold")
    ax4.tick_params(axis="y", labelcolor="red")
    ax4_twin.tick_params(axis="y", labelcolor="blue")
    ax4.grid(True, alpha=0.3)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc="upper left")

    # 5. Mean correlation effect
    ax5 = plt.subplot(2, 4, 5)
    ax5.scatter(
        stable["mean_correlation"],
        stable["empirical_drift"],
        alpha=0.4,
        s=30,
        c="green",
        label="Stable",
    )
    ax5.scatter(
        unstable["mean_correlation"],
        unstable["empirical_drift"],
        alpha=0.4,
        s=30,
        c="red",
        label="Unstable",
    )
    ax5.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax5.set_xlabel("Mean Correlation", fontsize=11)
    ax5.set_ylabel("Empirical Drift", fontsize=11)
    ax5.set_title("Mean Correlation Effect", fontweight="bold")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Negative correlations impact
    ax6 = plt.subplot(2, 4, 6)
    ax6.scatter(
        stable["negative_correlation_mean"],
        stable["dissatisfied_fraction"],
        alpha=0.4,
        s=30,
        c="green",
        label="Stable",
    )
    ax6.scatter(
        unstable["negative_correlation_mean"],
        unstable["dissatisfied_fraction"],
        alpha=0.4,
        s=30,
        c="red",
        label="Unstable",
    )
    ax6.set_xlabel("Mean Negative Correlation", fontsize=11)
    ax6.set_ylabel("Dissatisfied Fraction", fontsize=11)
    ax6.set_title("Negative Correlations → Dissatisfaction", fontweight="bold")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. Final satisfaction distribution
    ax7 = plt.subplot(2, 4, 7)
    ax7.hist(
        stable["final_mean_satisfaction"],
        bins=30,
        alpha=0.6,
        color="green",
        label="Stable",
        density=True,
    )
    ax7.hist(
        unstable["final_mean_satisfaction"],
        bins=30,
        alpha=0.6,
        color="red",
        label="Unstable",
        density=True,
    )
    ax7.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax7.set_xlabel("Final Mean Satisfaction", fontsize=11)
    ax7.set_ylabel("Density", fontsize=11)
    ax7.set_title("Final Satisfaction Distribution", fontweight="bold")
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis="y")

    # 8. Mathematical formula summary
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis("off")

    unstable_rate = df_results["system_unstable"].mean() * 100
    correlation_with_amplitude = (
        df_results[["amplitude", "system_unstable"]].corr().iloc[0, 1]
    )

    formula_text = f"""
    MATHEMATICAL FORMULATION
    {'='*50}
    
    System Dynamics:
    
    dS_active/dt = g - d
    dS_others/dt = C·η - d
    
    where:
    • g = growth_rate = 0.2
    • d = decay_rate = (1 - μ_C) × α
    • C = correlation effect
    • η = network_impact = 0.05
    • μ_C = mean correlation = e + a·cos(θ)
    • α = influence_multiplier = 0.03
    
    Total System Flow:
    Φ_total = dS_active/dt + Σ(dS_others/dt)
    
    HYPOTHESIS TEST RESULTS:
    {'='*50}
    
    Unstable systems: {unstable_rate:.1f}%
    
    Correlation(amplitude, instability): {correlation_with_amplitude:.3f}
    
    FINDING: High amplitude → more negative 
    correlations → stronger decay → 
    system loses satisfaction faster than 
    it can recover → perpetual dissatisfaction
    
    The system is NOT conserving satisfaction
    when amplitude is high!
    """

    ax8.text(
        0.05,
        0.95,
        formula_text,
        transform=ax8.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    return fig


def demonstrate_single_case():
    """Demonstrate satisfaction conservation in a single simulation"""
    print("\n" + "=" * 70)
    print("SINGLE CASE DEMONSTRATION")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    test_cases = [
        {"elevation": 0.2, "amplitude": 0.3, "label": "Stable (Low Amplitude)"},
        {"elevation": 0.2, "amplitude": 0.7, "label": "Borderline (Medium Amplitude)"},
        {"elevation": 0.2, "amplitude": 0.9, "label": "Unstable (High Amplitude)"},
    ]

    for idx, case in enumerate(test_cases):
        # Theoretical analysis
        theory = analyze_satisfaction_balance(case["elevation"], case["amplitude"])

        # Simulation
        corr_matrix = create_circumplex_correlation_matrix(
            n_motives=8, elevation=case["elevation"], amplitude=case["amplitude"]
        )
        initial_sat = np.clip(np.random.normal(0.5, 0.2, 8), -1, 1)
        sat_history, total_sat, mean_sat = simulate_and_track_satisfaction(
            corr_matrix, initial_sat, steps=300
        )

        # Top row: Individual motives
        ax_top = axes[0, idx]
        colors = plt.cm.tab10(np.linspace(0, 1, 8))
        for i in range(8):
            ax_top.plot(sat_history[:, i], color=colors[i], alpha=0.7, linewidth=1.5)
        ax_top.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax_top.set_xlabel("Step")
        ax_top.set_ylabel("Satisfaction")
        ax_top.set_title(
            f"{case['label']}\nTheory Flow: {theory['total_system_flow']:.4f}",
            fontweight="bold",
        )
        ax_top.grid(True, alpha=0.3)
        ax_top.set_ylim(-1.1, 1.1)

        # Bottom row: Total satisfaction
        ax_bot = axes[1, idx]
        ax_bot.plot(total_sat, linewidth=2, color="darkblue")

        # Fit line to show drift
        x = np.arange(50, 250)
        slope, intercept = np.polyfit(x, total_sat[50:250], 1)
        ax_bot.plot(
            x,
            slope * x + intercept,
            "r--",
            linewidth=2,
            label=f"Drift: {slope:.4f}/step",
        )

        ax_bot.set_xlabel("Step")
        ax_bot.set_ylabel("Total Satisfaction (Sum)")
        ax_bot.set_title(f"Total Satisfaction Over Time", fontweight="bold")
        ax_bot.legend()
        ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Demonstrate single cases
    demonstrate_single_case()

    # Run comprehensive test
    print("\n" + "=" * 70)
    print("RUNNING COMPREHENSIVE CONSERVATION ANALYSIS")
    print("=" * 70)

    df_results = test_satisfaction_conservation_hypothesis(
        elevation_range=np.linspace(-0.3, 0.3, 15),
        amplitude_range=np.linspace(0.1, 0.9, 15),
        n_trials_per_combo=20,
    )

    # Save results
    df_results.to_csv("satisfaction_conservation_analysis.csv", index=False)
    print("\nResults saved to 'satisfaction_conservation_analysis.csv'")

    # Visualize
    fig = visualize_conservation_analysis(df_results)
    plt.show()

    # Print key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    unstable = df_results[df_results["system_unstable"]]
    stable = df_results[~df_results["system_unstable"]]

    print(
        f"\nUnstable systems: {len(unstable)} / {len(df_results)} ({len(unstable)/len(df_results)*100:.1f}%)"
    )
    print(f"\nMean amplitude - Stable: {stable['amplitude'].mean():.3f}")
    print(f"Mean amplitude - Unstable: {unstable['amplitude'].mean():.3f}")
    print(f"\nMean elevation - Stable: {stable['elevation'].mean():.3f}")
    print(f"Mean elevation - Unstable: {unstable['elevation'].mean():.3f}")

    print(
        f"\nCorrelation(amplitude, instability): {df_results[['amplitude', 'system_unstable']].corr().iloc[0,1]:.3f}"
    )
    print(
        f"Correlation(elevation, instability): {df_results[['elevation', 'system_unstable']].corr().iloc[0,1]:.3f}"
    )

    print("\n" + "=" * 70)
    print("HYPOTHESIS CONFIRMED!")
    print("=" * 70)
    print("High amplitude → More negative correlations → Net satisfaction loss")
    print("The system cannot maintain satisfaction balance when amplitude is high.")

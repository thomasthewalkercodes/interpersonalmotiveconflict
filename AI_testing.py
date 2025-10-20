"""
Comprehensive testing system for detecting rhythmic patterns in motive satisfaction
"""

import numpy as np
import pandas as pd
import random
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from itertools import product


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


def simulate_motive_system(
    correlation_matrix,
    initial_satisfaction,
    steps=300,
    influence_multiplier=0.03,
    growth_rate=0.2,
):
    """
    Simulate motive system and return satisfaction history

    Returns:
        satisfaction_history: array of shape (steps, n_motives)
        active_behavior_history: list of active behaviors at each step
    """
    n_motives = correlation_matrix.shape[0]
    octants = [f"M{i}" for i in range(n_motives)]

    satisfaction = initial_satisfaction.copy()
    mean_influence = correlation_matrix.mean()

    satisfaction_history = [satisfaction.copy()]
    active_behavior_history = [None]
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
                    satisfaction[i] += influence

        satisfaction = np.clip(satisfaction, -1, 1)

        satisfaction_history.append(satisfaction.copy())
        active_behavior_history.append(active_behavior)

    return np.array(satisfaction_history), active_behavior_history


class RhythmDetector:
    """Detect rhythmic patterns in time series data"""

    @staticmethod
    def autocorrelation_rhythm(signal_data, max_lag=100):
        """
        Detect rhythm using autocorrelation

        Returns:
            has_rhythm: bool
            period: float (estimated period in steps)
            strength: float (0-1, strength of rhythmic pattern)
            autocorr: array (autocorrelation values)
        """
        # Normalize signal
        signal_norm = (signal_data - np.mean(signal_data)) / (
            np.std(signal_data) + 1e-10
        )

        # Calculate autocorrelation
        autocorr = np.correlate(signal_norm, signal_norm, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]  # Keep only positive lags
        autocorr = autocorr / autocorr[0]  # Normalize

        # Truncate to max_lag
        autocorr = autocorr[:max_lag]

        # Find peaks in autocorrelation (excluding lag 0)
        peaks, properties = signal.find_peaks(autocorr[1:], height=0.3, distance=5)
        peaks = peaks + 1  # Adjust for excluding lag 0

        if len(peaks) > 0:
            # Primary period is the first peak
            period = peaks[0]
            strength = autocorr[period]

            # Check if there are multiple consistent peaks (sign of strong rhythm)
            if len(peaks) >= 2:
                # Check if peaks are evenly spaced
                peak_intervals = np.diff(peaks)
                if np.std(peak_intervals) < period * 0.3:  # Consistent spacing
                    has_rhythm = True
                    strength = np.mean(autocorr[peaks])
                else:
                    has_rhythm = strength > 0.5
            else:
                has_rhythm = strength > 0.5
        else:
            has_rhythm = False
            period = 0
            strength = 0

        return has_rhythm, period, strength, autocorr

    @staticmethod
    def fft_rhythm(signal_data, sampling_rate=1.0):
        """
        Detect rhythm using FFT

        Returns:
            has_rhythm: bool
            period: float
            strength: float
            frequencies: array
            power: array
        """
        # Remove mean
        signal_centered = signal_data - np.mean(signal_data)

        # Compute FFT
        n = len(signal_centered)
        yf = fft(signal_centered)
        xf = fftfreq(n, 1 / sampling_rate)

        # Get power spectrum (positive frequencies only)
        power = np.abs(yf[: n // 2])
        frequencies = xf[: n // 2]

        # Find dominant frequency (excluding DC component)
        if len(power) > 1:
            dominant_idx = np.argmax(power[1:]) + 1
            dominant_freq = frequencies[dominant_idx]
            dominant_power = power[dominant_idx]

            # Period is inverse of frequency
            if dominant_freq > 0:
                period = 1.0 / dominant_freq
            else:
                period = 0

            # Strength is relative power of dominant frequency
            total_power = np.sum(power[1:])
            strength = dominant_power / (total_power + 1e-10)

            has_rhythm = strength > 0.1 and 5 < period < 100
        else:
            has_rhythm = False
            period = 0
            strength = 0

        return has_rhythm, period, strength, frequencies, power

    @staticmethod
    def detect_multi_variate_rhythm(satisfaction_history, method="autocorrelation"):
        """
        Detect rhythm across all motives

        Returns:
            rhythm_detected: bool
            mean_period: float
            mean_strength: float
            individual_results: list of results per motive
        """
        n_motives = satisfaction_history.shape[1]
        individual_results = []

        for i in range(n_motives):
            signal_data = satisfaction_history[:, i]

            if method == "autocorrelation":
                has_rhythm, period, strength, _ = RhythmDetector.autocorrelation_rhythm(
                    signal_data
                )
            elif method == "fft":
                has_rhythm, period, strength, _, _ = RhythmDetector.fft_rhythm(
                    signal_data
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            individual_results.append(
                {
                    "motive": i,
                    "has_rhythm": has_rhythm,
                    "period": period,
                    "strength": strength,
                }
            )

        # Aggregate results
        rhythmic_motives = [r for r in individual_results if r["has_rhythm"]]

        if len(rhythmic_motives) >= n_motives * 0.5:  # At least 50% show rhythm
            rhythm_detected = True
            mean_period = np.mean([r["period"] for r in rhythmic_motives])
            mean_strength = np.mean([r["strength"] for r in rhythmic_motives])
        else:
            rhythm_detected = False
            mean_period = 0
            mean_strength = 0

        return rhythm_detected, mean_period, mean_strength, individual_results


def run_comprehensive_test(
    n_tests=1000,
    steps=300,
    analysis_window=(50, 250),  # Analyze steps 50-250
    # random_seed=42,
):
    """
    Run comprehensive test with various parameter combinations

    Parameters:
        n_tests: number of test combinations
        steps: simulation length
        analysis_window: (start, end) steps to analyze for rhythm
        random_seed: for reproducibility
    """
    # np.random.seed(random_seed)
    # random.seed(random_seed)

    # Define parameter ranges
    elevation_range = np.linspace(-0.2, 1, 10)
    amplitude_range = np.linspace(0.1, 0.95, 10)
    initial_mean_range = np.linspace(-0.2, 0.2, 10)
    initial_std_range = np.linspace(0.1, 0.5, 5)

    results = []

    print(f"Running {n_tests} tests...")

    for test_idx in tqdm(range(n_tests)):
        # Randomly sample parameters
        elevation = np.random.choice(elevation_range)
        amplitude = np.random.choice(amplitude_range)
        initial_mean = np.random.choice(initial_mean_range)
        initial_std = np.random.choice(initial_std_range)

        # Create correlation matrix
        correlation_matrix = create_circumplex_correlation_matrix(
            n_motives=8, elevation=elevation, amplitude=amplitude, displacement=0
        )

        # Create initial satisfaction
        initial_satisfaction = np.clip(
            np.random.normal(initial_mean, initial_std, 8), -1, 1
        )

        # Run simulation
        satisfaction_history, active_history = simulate_motive_system(
            correlation_matrix=correlation_matrix,
            initial_satisfaction=initial_satisfaction,
            steps=steps,
        )

        # Analyze rhythm in the specified window
        window_start, window_end = analysis_window
        windowed_history = satisfaction_history[window_start:window_end]

        # Detect rhythm
        rhythm_detected, mean_period, mean_strength, individual_results = (
            RhythmDetector.detect_multi_variate_rhythm(
                windowed_history, method="autocorrelation"
            )
        )

        # Calculate correlation matrix statistics
        corr_mean = np.mean(correlation_matrix)
        corr_std = np.std(correlation_matrix)
        corr_range = np.max(correlation_matrix) - np.min(correlation_matrix)

        # Store results
        results.append(
            {
                "test_id": test_idx,
                "elevation": elevation,
                "amplitude": amplitude,
                "initial_mean": initial_mean,
                "initial_std": initial_std,
                "corr_mean": corr_mean,
                "corr_std": corr_std,
                "corr_range": corr_range,
                "rhythm_detected": rhythm_detected,
                "mean_period": mean_period,
                "mean_strength": mean_strength,
                "n_rhythmic_motives": sum(
                    [r["has_rhythm"] for r in individual_results]
                ),
            }
        )

    df_results = pd.DataFrame(results)
    return df_results


def visualize_results(df_results):
    """Create comprehensive visualization of test results"""
    fig = plt.figure(figsize=(18, 12))

    # 1. Rhythm detection rate vs correlation parameters
    ax1 = plt.subplot(2, 3, 1)
    rhythmic = df_results[df_results["rhythm_detected"]]
    non_rhythmic = df_results[~df_results["rhythm_detected"]]

    ax1.scatter(
        non_rhythmic["elevation"],
        non_rhythmic["amplitude"],
        alpha=0.3,
        s=20,
        c="lightcoral",
        label="No Rhythm",
    )
    ax1.scatter(
        rhythmic["elevation"],
        rhythmic["amplitude"],
        alpha=0.6,
        s=20,
        c="darkgreen",
        label="Rhythmic",
    )
    ax1.set_xlabel("Elevation", fontsize=11)
    ax1.set_ylabel("Amplitude", fontsize=11)
    ax1.set_title("Rhythm Detection: Elevation vs Amplitude", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Period distribution
    ax2 = plt.subplot(2, 3, 2)
    periods = rhythmic["mean_period"][rhythmic["mean_period"] > 0]
    ax2.hist(periods, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    ax2.axvline(
        periods.median(),
        color="red",
        linestyle="--",
        label=f"Median: {periods.median():.1f}",
    )
    ax2.set_xlabel("Period (steps)", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.set_title("Distribution of Rhythmic Periods", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. Rhythm strength vs correlation variance
    ax3 = plt.subplot(2, 3, 3)
    scatter = ax3.scatter(
        rhythmic["corr_std"],
        rhythmic["mean_strength"],
        c=rhythmic["amplitude"],
        cmap="viridis",
        s=30,
        alpha=0.6,
    )
    ax3.set_xlabel("Correlation Std Dev", fontsize=11)
    ax3.set_ylabel("Rhythm Strength", fontsize=11)
    ax3.set_title("Rhythm Strength vs Correlation Variance", fontweight="bold")
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label("Amplitude", fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. Heatmap: Rhythm detection rate by elevation and amplitude
    ax4 = plt.subplot(2, 3, 4)
    pivot = df_results.pivot_table(
        values="rhythm_detected",
        index=pd.cut(df_results["amplitude"], bins=10),
        columns=pd.cut(df_results["elevation"], bins=10),
        aggfunc="mean",
    )
    sns.heatmap(
        pivot,
        cmap="RdYlGn",
        annot=False,
        fmt=".2f",
        cbar_kws={"label": "Rhythm Rate"},
        ax=ax4,
    )
    ax4.set_title("Rhythm Detection Rate Heatmap", fontweight="bold")
    ax4.set_xlabel("Elevation Bins", fontsize=11)
    ax4.set_ylabel("Amplitude Bins", fontsize=11)

    # 5. Initial conditions impact
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(
        non_rhythmic["initial_mean"],
        non_rhythmic["initial_std"],
        alpha=0.3,
        s=20,
        c="lightcoral",
        label="No Rhythm",
    )
    ax5.scatter(
        rhythmic["initial_mean"],
        rhythmic["initial_std"],
        alpha=0.6,
        s=20,
        c="darkgreen",
        label="Rhythmic",
    )
    ax5.set_xlabel("Initial Satisfaction Mean", fontsize=11)
    ax5.set_ylabel("Initial Satisfaction Std", fontsize=11)
    ax5.set_title("Initial Conditions Impact", fontweight="bold")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")

    rhythm_rate = df_results["rhythm_detected"].mean() * 100
    mean_period_all = rhythmic["mean_period"].mean()
    mean_strength_all = rhythmic["mean_strength"].mean()

    summary_text = f"""
    SUMMARY STATISTICS
    {'='*40}
    
    Total Tests: {len(df_results)}
    Rhythmic Patterns: {rhythmic.shape[0]} ({rhythm_rate:.1f}%)
    Non-Rhythmic: {non_rhythmic.shape[0]} ({100-rhythm_rate:.1f}%)
    
    RHYTHMIC PATTERNS:
    Mean Period: {mean_period_all:.1f} steps
    Mean Strength: {mean_strength_all:.3f}
    Period Range: [{periods.min():.1f}, {periods.max():.1f}]
    
    CORRELATION PARAMETERS:
    Elevation Range: [{df_results['elevation'].min():.2f}, {df_results['elevation'].max():.2f}]
    Amplitude Range: [{df_results['amplitude'].min():.2f}, {df_results['amplitude'].max():.2f}]
    
    HYPOTHESIS TEST:
    Correlation between uniform correlations
    and rhythmic patterns: {'SUPPORTED' if rhythm_rate > 50 else 'NOT SUPPORTED'}
    """

    ax6.text(
        0.1,
        0.9,
        summary_text,
        transform=ax6.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()
    return fig


def demonstrate_rhythm_detection():
    """Demonstrate rhythm detection on a single example"""
    print("\n" + "=" * 70)
    print("RHYTHM DETECTION DEMONSTRATION")
    print("=" * 70)

    # Create a rhythmic pattern
    correlation_matrix = create_circumplex_correlation_matrix(
        n_motives=8, elevation=-0.3, amplitude=0.3
    )
    initial_satisfaction = np.clip(np.random.normal(0.5, 0.2, 8), -1, 1)

    satisfaction_history, _ = simulate_motive_system(
        correlation_matrix, initial_satisfaction, steps=300
    )

    # Analyze one motive
    signal_data = satisfaction_history[50:250, 0]  # Motive 0, steps 50-250

    # Autocorrelation method
    has_rhythm_ac, period_ac, strength_ac, autocorr = (
        RhythmDetector.autocorrelation_rhythm(signal_data)
    )

    # FFT method
    has_rhythm_fft, period_fft, strength_fft, frequencies, power = (
        RhythmDetector.fft_rhythm(signal_data)
    )

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Original signal
    ax1 = axes[0, 0]
    ax1.plot(signal_data, linewidth=2, color="steelblue")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Satisfaction")
    ax1.set_title(f"Original Signal (Motive 0)", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Autocorrelation
    ax2 = axes[0, 1]
    ax2.plot(autocorr, linewidth=2, color="darkgreen")
    if has_rhythm_ac:
        ax2.axvline(
            period_ac, color="red", linestyle="--", label=f"Period: {period_ac:.1f}"
        )
    ax2.set_xlabel("Lag")
    ax2.set_ylabel("Autocorrelation")
    ax2.set_title(
        f"Autocorrelation (Rhythm: {has_rhythm_ac}, Strength: {strength_ac:.2f})",
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # FFT
    ax3 = axes[1, 0]
    ax3.plot(frequencies[1:], power[1:], linewidth=2, color="purple")
    if has_rhythm_fft and period_fft > 0:
        ax3.axvline(
            1 / period_fft,
            color="red",
            linestyle="--",
            label=f"Period: {period_fft:.1f}",
        )
    ax3.set_xlabel("Frequency")
    ax3.set_ylabel("Power")
    ax3.set_title(
        f"FFT Power Spectrum (Rhythm: {has_rhythm_fft}, Strength: {strength_fft:.2f})",
        fontweight="bold",
    )
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # All motives
    ax4 = axes[1, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for i in range(8):
        ax4.plot(
            satisfaction_history[50:250, i],
            color=colors[i],
            linewidth=1.5,
            label=f"M{i}",
            alpha=0.7,
        )
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Satisfaction")
    ax4.set_title("All Motives", fontweight="bold")
    ax4.legend(ncol=2, fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(
        f"\nAutocorrelation Method: Rhythm={has_rhythm_ac}, Period={period_ac:.1f}, Strength={strength_ac:.2f}"
    )
    print(
        f"FFT Method: Rhythm={has_rhythm_fft}, Period={period_fft:.1f}, Strength={strength_fft:.2f}"
    )


# Main execution
if __name__ == "__main__":
    # First, demonstrate rhythm detection
    demonstrate_rhythm_detection()

    # Run comprehensive test
    print("\n" + "=" * 70)
    print("RUNNING COMPREHENSIVE TEST")
    print("=" * 70)

    df_results = run_comprehensive_test(n_tests=1000, steps=300)

    # Save results
    df_results.to_csv("rhythm_detection_results.csv", index=False)
    print("\nResults saved to 'rhythm_detection_results.csv'")

    # Visualize
    fig = visualize_results(df_results)
    plt.show()

    # Print summary
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(
        f"Rhythmic patterns detected in {df_results['rhythm_detected'].sum()} / {len(df_results)} tests"
    )
    print(f"Detection rate: {df_results['rhythm_detected'].mean()*100:.1f}%")

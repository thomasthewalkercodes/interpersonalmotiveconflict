import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_circumplex_correlation_matrix(
    n_motives=8, elevation=0.1, amplitude=0.9, displacement=0.0
):
    """
    Create a correlation matrix based on circumplex structure using cosine wave.

    Formula: r_i^p = e + a * cos(θ_i - δ)

    Parameters:
    -----------
    n_motives : int
        Number of motives/octants (default: 8)
    elevation : float
        Baseline correlation level (e parameter)
    amplitude : float
        Maximum deviation from elevation (a parameter)
    displacement : float
        Angular displacement in degrees (δ parameter)

    Returns:
    --------
    pd.DataFrame
        Correlation matrix with motive labels
    """
    # Initialize matrix
    correlation_matrix = np.eye(n_motives)  # Diagonal is 1 (self-correlation)

    # Calculate angle for each motive (in degrees)
    angles = np.array([i * (360 / n_motives) for i in range(n_motives)])

    # Convert displacement to radians for calculation
    displacement_rad = np.radians(displacement)

    # Calculate correlations for all pairs
    for i in range(n_motives):
        for j in range(n_motives):
            if i != j:
                # Calculate angular difference
                angle_i = np.radians(angles[i])
                angle_j = np.radians(angles[j])

                # Angular distance (absolute difference)
                angle_diff = angle_i - angle_j

                # Apply cosine formula: r = e + a * cos(θ - δ)
                correlation = elevation + amplitude * np.cos(
                    angle_diff - displacement_rad
                )

                # Ensure symmetric matrix
                correlation_matrix[i, j] = correlation

    # Create DataFrame with labels
    labels = ["LM", "NO", "AP", "BC", "DE", "FG", "HI", "JK"]
    df_correlation = pd.DataFrame(correlation_matrix, index=labels, columns=labels)

    return df_correlation


def visualize_correlation_matrix(
    correlation_matrix, title="Circumplex Correlation Matrix"
):
    """Visualize the correlation matrix as a heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(correlation_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)

    # Set ticks and labels
    ax.set_xticks(range(len(correlation_matrix)))
    ax.set_yticks(range(len(correlation_matrix)))
    ax.set_xticklabels(correlation_matrix.columns)
    ax.set_yticklabels(correlation_matrix.index)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation", rotation=270, labelpad=20)

    # Add correlation values as text
    for i in range(len(correlation_matrix)):
        for j in range(len(correlation_matrix)):
            text = ax.text(
                j,
                i,
                f"{correlation_matrix.values[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Motive", fontsize=12)
    ax.set_ylabel("Motive", fontsize=12)

    plt.tight_layout()
    return fig


def visualize_circumplex_profile(elevation, amplitude, displacement=0.5, n_points=360):
    """
    Visualize the circumplex correlation profile as a function of angular distance.

    This shows how correlation changes with angle according to: r = e + a * cos(θ - δ)
    """
    angles = np.linspace(0, 360, n_points)
    angles_rad = np.radians(angles)
    displacement_rad = np.radians(displacement)

    correlations = elevation + amplitude * np.cos(angles_rad - displacement_rad)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(angles, correlations, linewidth=2, color="darkblue")
    ax.axhline(
        y=elevation,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=f"Elevation = {elevation}",
    )
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    # Mark the 8 octant positions
    octant_angles = [i * 45 for i in range(8)]
    octant_correlations = elevation + amplitude * np.cos(
        np.radians(octant_angles) - displacement_rad
    )
    ax.scatter(
        octant_angles,
        octant_correlations,
        color="red",
        s=100,
        zorder=5,
        label="Octant positions",
    )

    ax.set_xlabel("Angular Distance (degrees)", fontsize=12)
    ax.set_ylabel("Predicted Correlation", fontsize=12)
    ax.set_title("Circumplex Correlation Profile", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 360)

    plt.tight_layout()
    return fig


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Circumplex Correlation Matrix Generator")
    print("=" * 60)

    # Example 1: Default parameters (similar to your covariance_matrix.py)
    print("\n1. Default configuration (elevation=0.1, amplitude=0.9):")
    corr_matrix_1 = create_circumplex_correlation_matrix(
        n_motives=8, elevation=0.1, amplitude=0.9, displacement=0.0
    )
    print(corr_matrix_1.round(3))

    # Example 2: Higher elevation (more positive baseline)
    print("\n2. Higher elevation (elevation=0.3, amplitude=0.7):")
    corr_matrix_2 = create_circumplex_correlation_matrix(
        n_motives=8, elevation=0.3, amplitude=0.7, displacement=0.0
    )
    print(corr_matrix_2.round(3))

    # Example 3: With angular displacement
    print("\n3. With displacement (elevation=0.0, amplitude=0.8, displacement=22.5°):")
    corr_matrix_3 = create_circumplex_correlation_matrix(
        n_motives=8, elevation=0.0, amplitude=0.8, displacement=22.5
    )
    print(corr_matrix_3.round(3))

    # Visualizations
    print("\nGenerating visualizations...")

    # Visualize correlation matrix
    fig1 = visualize_correlation_matrix(
        corr_matrix_1, "Circumplex Correlation Matrix (e=0.1, a=0.9)"
    )

    # Visualize circumplex profile
    fig2 = visualize_circumplex_profile(elevation=0.1, amplitude=0.9, displacement=0.0)

    plt.show()

    print("\n" + "=" * 60)
    print("Matrix generation complete!")
    print("=" * 60)

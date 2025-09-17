# Here I can use the sinus curve thingy and then
# calculate it.
import numpy as np

np.random.seed(42)
n = 8
num_matrices = 100
mu_values = np.linspace(-0.1, 0.2, num_matrices)  # Fixed typo: linspace not linespace
sigma = 0.1


def create_symmetric_matrix(n, mu=0, sigma=1):
    mat = np.zeros((n, n))
    # generate upper triangular matrix
    upper_indices = np.triu_indices(n, k=1)
    off_diag_values = np.random.normal(
        mu, sigma, size=len(upper_indices[0])
    )  # gaussian distribution with mean mu and SD sigma
    mat[upper_indices] = off_diag_values
    # mirror it
    mat = mat + mat.T
    # set diagonal to 1
    np.fill_diagonal(mat, 1)
    return mat


# generate matrices
matrices = [create_symmetric_matrix(n=n, mu=mu, sigma=sigma) for mu in mu_values]


def create_sinusoidal_covariance_matrix(n_octants=8, amplitude=1.0, elevation=0.0):
    """
    Create a covariance matrix based on sinusoidal relationships between octants.

    Args:
        n_octants: Number of octants (default 8)
        amplitude: Amplitude of the sinusoidal wave (controls strength of covariance)
        elevation: Elevation/offset of the sinusoidal wave (baseline covariance)

    Returns:
        Covariance matrix where adjacent octants have high covariance and
        opposing octants have negative covariance
    """
    covariance_matrix = np.zeros((n_octants, n_octants))

    for i in range(n_octants):
        for j in range(n_octants):
            if i == j:
                covariance_matrix[i, j] = 1.0  # Self-covariance is 1
            else:
                # Calculate angular distance between octants
                # Each octant is 2π/8 = π/4 radians apart
                angle_i = i * (2 * np.pi / n_octants)
                angle_j = j * (2 * np.pi / n_octants)

                # Calculate the minimum angular difference (accounting for circular nature)
                angle_diff = abs(angle_i - angle_j)
                angle_diff = min(angle_diff, 2 * np.pi - angle_diff)

                # Create sinusoidal covariance
                # cos(0) = 1 (adjacent), cos(π) = -1 (opposite)
                covariance_matrix[i, j] = amplitude * np.cos(angle_diff) + elevation

    return covariance_matrix


class MotiveSystem:
    def __init__(
        self,
        initial_values,
        decay_rate=0.05,
        growth_rate=0.3,
        max_satisfaction=1.0,
        covariance_amplitude=0.5,
        covariance_elevation=0.0,
        covariance_strength=0.02,
    ):
        """
        Initialize motive system with sinusoidal covariance

        Args:
            initial_values: Initial satisfaction levels for each motive (8 values)
            decay_rate: Base rate of satisfaction decrease each round
            growth_rate: How much active motive increases each round
            max_satisfaction: Maximum satisfaction level
            covariance_amplitude: Amplitude of sinusoidal covariance
            covariance_elevation: Elevation/baseline of sinusoidal covariance
            covariance_strength: How much covariance affects satisfaction changes
        """
        self.motives = np.array(initial_values)
        self.decay_rate = decay_rate
        self.growth_rate = growth_rate
        self.max_satisfaction = max_satisfaction
        self.active_motive = None
        self.history = [self.motives.copy()]

        # Covariance parameters
        self.covariance_amplitude = covariance_amplitude
        self.covariance_elevation = covariance_elevation
        self.covariance_strength = covariance_strength

        # Create covariance matrix
        self.covariance_matrix = create_sinusoidal_covariance_matrix(
            n_octants=len(initial_values),
            amplitude=covariance_amplitude,
            elevation=covariance_elevation,
        )

    def calculate_covariance_effects(self):
        """
        Calculate how covariance with neighboring octants affects each motive's change

        Returns:
            Array of covariance effects for each motive
        """
        effects = np.zeros(len(self.motives))

        for i in range(len(self.motives)):
            # Calculate weighted influence from all other motives
            covariance_influence = 0
            for j in range(len(self.motives)):
                if i != j:
                    # Influence = covariance * neighbor's satisfaction level
                    covariance_influence += (
                        self.covariance_matrix[i, j] * self.motives[j]
                    )

            # Scale by covariance strength
            effects[i] = covariance_influence * self.covariance_strength

        return effects

    def update_round(self):
        """Update motives for one round with covariance effects"""
        # Calculate covariance effects
        covariance_effects = self.calculate_covariance_effects()

        # Step 1: Apply decay with covariance modulation
        # Positive covariance effects reduce decay, negative increase it
        for i in range(len(self.motives)):
            effective_decay = self.decay_rate - covariance_effects[i]
            self.motives[i] -= max(0, effective_decay)  # Don't let decay go negative

            # Enforce minimum satisfaction level of -1
            self.motives[i] = max(-1.0, self.motives[i])

        # Step 2: If there's an active motive, grow it rapidly
        if self.active_motive is not None:
            self.motives[self.active_motive] += self.growth_rate

            # Check if active motive reached max satisfaction
            if self.motives[self.active_motive] >= self.max_satisfaction:
                self.active_motive = None  # Deactivate it

        # Step 3: If no active motive, choose a dissatisfied one
        if self.active_motive is None:
            dissatisfied = np.where(self.motives < 0)[0]

            if len(dissatisfied) > 0:
                if len(dissatisfied) == 1:
                    # Only one dissatisfied motive
                    self.active_motive = dissatisfied[0]
                else:
                    # Multiple dissatisfied motives - choose based on how dissatisfied they are
                    dissatisfaction_levels = -self.motives[
                        dissatisfied
                    ]  # More negative = more dissatisfied
                    probabilities = (
                        dissatisfaction_levels / dissatisfaction_levels.sum()
                    )
                    self.active_motive = np.random.choice(dissatisfied, p=probabilities)

        # Store history
        self.history.append(self.motives.copy())

    def get_satisfied_motives(self):
        """Return indices of satisfied motives (>0)"""
        return np.where(self.motives > 0)[0]

    def get_dissatisfied_motives(self):
        """Return indices of dissatisfied motives (<0)"""
        return np.where(self.motives < 0)[0]

    def simulate_rounds(self, num_rounds):
        """Simulate multiple rounds and track active motives"""
        self.active_motive_history = [self.active_motive]
        for _ in range(num_rounds):
            self.update_round()
            self.active_motive_history.append(self.active_motive)
        return np.array(self.history)

    def print_covariance_info(self):
        """Print information about the covariance matrix"""
        print("Covariance Matrix:")
        print(self.covariance_matrix.round(3))
        print(f"\nCovariance Parameters:")
        print(f"  Amplitude: {self.covariance_amplitude}")
        print(f"  Elevation: {self.covariance_elevation}")
        print(f"  Strength: {self.covariance_strength}")


# Example usage
if __name__ == "__main__":
    # Create initial motive values
    initial_motives = np.random.normal(0, 0.5, 8)

    # Create motive system with covariance
    motive_system = MotiveSystem(
        initial_motives,
        decay_rate=0.05,
        growth_rate=0.3,
        max_satisfaction=1.0,
        covariance_amplitude=1,  # Strong covariance effect
        covariance_elevation=0,  # Slight positive baseline
        covariance_strength=0.03,  # Moderate influence on satisfaction changes
    )

    print("Initial motive levels:", motive_system.motives.round(3))
    motive_system.print_covariance_info()

    # Simulate 30 rounds
    history = motive_system.simulate_rounds(300)

    print(f"\nAfter 30 rounds:")
    print("Final motive levels:", motive_system.motives.round(3))
    print("Active motive:", motive_system.active_motive)

    # Visualize the results
    import matplotlib.pyplot as plt

    # Create figure with better screen fitting layout
    fig = plt.figure(figsize=(16, 9))  # 16:9 aspect ratio for better screen fit

    # Create subplot layout: 2x2 grid
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)  # Top row spans both columns
    ax2 = plt.subplot2grid((2, 2), (1, 0))  # Bottom left
    ax3 = plt.subplot2grid((2, 2), (1, 1))  # Bottom right

    # Plot 1: Motive evolution over time
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for i in range(8):
        ax1.plot(history[:, i], label=f"Octant {i}", color=colors[i], linewidth=2)
    ax1.axhline(y=0, color="black", linestyle="--", alpha=0.7, linewidth=1)
    ax1.axhline(
        y=1, color="red", linestyle="--", alpha=0.7, linewidth=1, label="Max (1.0)"
    )
    ax1.axhline(
        y=-1, color="red", linestyle="--", alpha=0.7, linewidth=1, label="Min (-1.0)"
    )
    ax1.set_xlabel("Round", fontsize=12)
    ax1.set_ylabel("Satisfaction Level", fontsize=12)
    ax1.set_title("Motive Satisfaction Over Time", fontsize=14, fontweight="bold")
    ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.1, 1.1)

    # Plot 2: Active motive matrix-style visualization
    # Create a matrix where rows are motives (0-7) and columns are rounds
    active_matrix = np.full(
        (8, len(motive_system.active_motive_history)), -1, dtype=float
    )

    for round_idx, active_motive in enumerate(motive_system.active_motive_history):
        if active_motive is not None:
            active_matrix[active_motive, round_idx] = 1  # Active motive gets value 1
        # All other motives remain -1 (inactive)

    # Create custom colormap: white for inactive (-1), colored for active (1)
    from matplotlib.colors import ListedColormap

    colors_matrix = ["lightgray", "darkred"]  # Gray for inactive, red for active
    cmap_custom = ListedColormap(colors_matrix)

    im2 = ax2.imshow(active_matrix, cmap=cmap_custom, aspect="auto", vmin=-1, vmax=1)
    ax2.set_title("Active Motive Matrix", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Round", fontsize=12)
    ax2.set_ylabel("Octant", fontsize=12)
    ax2.set_yticks(range(8))
    ax2.set_yticklabels([f"Octant {i}" for i in range(8)], fontsize=10)

    # Add grid lines
    ax2.set_xticks(
        np.arange(-0.5, len(motive_system.active_motive_history), 1), minor=True
    )
    ax2.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax2.grid(which="minor", color="white", linestyle="-", linewidth=0.5)

    # Plot 3: Covariance matrix as heatmap
    im3 = ax3.imshow(motive_system.covariance_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax3.set_title("Covariance Matrix", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Octant", fontsize=12)
    ax3.set_ylabel("Octant", fontsize=12)
    ax3.set_xticks(range(8))
    ax3.set_yticks(range(8))
    ax3.set_xticklabels([f"{i}" for i in range(8)], fontsize=10)
    ax3.set_yticklabels([f"{i}" for i in range(8)], fontsize=10)

    # Add colorbar for covariance matrix
    cbar = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar.set_label("Covariance", fontsize=11)

    # Add text annotations to covariance matrix
    for i in range(8):
        for j in range(8):
            text = ax3.text(
                j,
                i,
                f"{motive_system.covariance_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    plt.tight_layout()
    plt.show()

# Here I can use the sinus curve thingy and then
# calculate it.
import numpy as np

np.random.seed(42)
n = 8
num_matrices = 100
mu_values = np.linespace(-0.1, 0.2, num_matrices)
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


class MotiveSystem:
    def __init__(
        self, initial_values, decay_rate=0.05, growth_rate=0.3, max_satisfaction=4.0
    ):
        """
        Initialize motive system

        Args:
            initial_values: Initial satisfaction levels for each motive (8 values)
            decay_rate: How much satisfaction decreases each round
            growth_rate: How much active motive increases each round
            max_satisfaction: Maximum satisfaction level
        """
        self.motives = np.array(initial_values)
        self.decay_rate = decay_rate
        self.growth_rate = growth_rate
        self.max_satisfaction = max_satisfaction
        self.active_motive = None
        self.history = [self.motives.copy()]

    def update_round(self):
        """Update motives for one round"""
        # Step 1: All motives lose satisfaction slightly
        self.motives -= self.decay_rate

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
        """Simulate multiple rounds"""
        for _ in range(num_rounds):
            self.update_round()
        return np.array(self.history)

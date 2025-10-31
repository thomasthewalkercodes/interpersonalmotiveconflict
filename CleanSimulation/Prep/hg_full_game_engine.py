# Imports
import numpy as np
import pandas as pd
import random

# game_engine, and then all the rest
def game_engine(sat_m, inter_m, steps, decay_rate, growth_rate=1):
    history = {
        "step": [],
        "active_behavior": [],
        "satisfaction": [],
        "unsatisfied_octants": [],
    }

    active_behavior = None

    for step in range(steps):
        # Get current satisfaction levels
        satisfaction_levels = sat_m.loc["satisfaction"]
        unsatisfied_octants = satisfaction_levels[
            satisfaction_levels < 0
        ].index.tolist()

        # Switch behavior of current one is satisfied (reached 1) or none is active
        if active_behavior is None or satisfaction_levels[active_behavior] >= 1:
            active_behavior = select_behavior(satisfaction_levels, unsatisfied_octants)

        # Apply growth and influence if a behavior is active
        if active_behavior is not None:
            # active behavior grows
            sat_m.loc["satisfaction", active_behavior] += growth_rate

            # Active behavior influences others (not itself)
            for octant in sat_m.columns:
                if octant != active_behavior:
                    influence = inter_m.loc[active_behavior, octant]
                    sat_m.loc["satisfaction", octant] += influence

        # Apply decay AFTER growth and influence (not the active motive)
        for octant in sat_m.columns:
            if octant != active_behavior:
                sat_m.loc["satisfaction", octant] -= decay_rate / 8
        # Clip all satisfaction values to be within the range [-1, 1]
        sat_m.loc["satisfaction"] = np.clip(sat_m.loc["satisfaction"], -1, 1)

        # Record history (after all updates)
        history["step"].append(step)
        history["active_behavior"].append(active_behavior)
        history["satisfaction"].append(sat_m.copy())
        history["unsatisfied_octants"].append(unsatisfied_octants)

    return history

def select_behavior(satisfaction_levels, unsatisfied_octants):
    if not unsatisfied_octants:
        return None
    probs = -satisfaction_levels[unsatisfied_octants]
    probs = probs / probs.sum()  # Normalize to sum to 1
    return random.choices(unsatisfied_octants, weights=probs, k=1)[0]

def generate_satisfaction_matrix(n_motives=8, mean=0.3, sd=0.5):
    """Generate initial satisfaction matrix."""
    sat_values = np.random.normal(mean, sd, size=n_motives)
    sat_values = np.clip(sat_values, -1, 1)
    sat_m = pd.DataFrame(
        [sat_values],
        columns=[f"motive_{i+1}" for i in range(n_motives)],
        index=["satisfaction"],
    )
    return sat_m

def generate_interaction_matrix(n_motives=8, mean=0.0, sd=0.2):
    """Generate symmetric interaction matrix."""
    # Create a random interaction matrix
    matrix = np.random.normal(mean, sd, size=(n_motives, n_motives))
    matrix = (matrix + matrix.T) / 2  # Make it symmetric
    np.fill_diagonal(matrix, 0)  # No self-influence
    return pd.DataFrame(
        matrix,
        columns=[f"motive_{i+1}" for i in range(n_motives)],
        index=[f"motive_{i+1}" for i in range(n_motives)],
    )

def generate_ind_decay_rate(n_motives = 8, mean =):

    


def get_lambda(inter_m, decay_lambda=None):
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

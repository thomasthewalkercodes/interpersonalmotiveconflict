import random
import pandas as pd

steps = 50
decay_rate = 0.1


def select_behavior(satisfaction_levels, unsatisfied_octants):
    if not unsatisfied_octants:
        return None
    probs = -satisfaction_levels[unsatisfied_octants]
    probs = probs / probs.sum()  # Normalize to sum to 1
    return random.choices(unsatisfied_octants, weights=probs, k=1)[0]


def game_engine(
    satisfaction_matrix, influence_matrix, steps, decay_rate, growth_rate=1
):
    history = {
        "step": [],
        "active_behavior": [],
        "satisfaction": [],
        "unsatisfied_octants": [],
    }

    active_behavior = None

    for step in range(steps):
        # Get current satisfaction levels
        satisfaction_levels = satisfaction_matrix.loc["satisfaction"]
        unsatisfied_octants = satisfaction_levels[
            satisfaction_levels < 0
        ].index.tolist()

        # Switch behavior of current one is satisfied (reached 1) or none is active
        if active_behavior is None or satisfaction_levels[active_behavior] >= 1:
            active_behavior = select_behavior(satisfaction_levels, unsatisfied_octants)

        # Apply growth and influence if a behavior is active
        if active_behavior is not None:
            # active behavior grows
            satisfaction_matrix.loc["satisfaction", active_behavior] += growth_rate

            # Active behavior influences others (not itself)
            for octant in satisfaction_matrix.columns:
                if octant != active_behavior:
                    influence = influence_matrix.loc[active_behavior, octant]
                    satisfaction_matrix.loc["satisfaction", octant] += influence

        # Apply decay AFTER growth and influence
        for octant in satisfaction_matrix.columns:
            if octant != active_behavior:
                satisfaction_matrix.loc["satisfaction", octant] -= decay_rate

        # Record history (after all updates)
        history["step"].append(step)
        history["active_behavior"].append(active_behavior)
        history["satisfaction"].append(satisfaction_matrix.copy())
        history["unsatisfied_octants"].append(unsatisfied_octants)

    return history

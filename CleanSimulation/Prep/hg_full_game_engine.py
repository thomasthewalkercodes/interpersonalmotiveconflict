# Imports
import numpy as np
import pandas as pd
import random

from decay_matrix import GenerateDecayMatrix


# game_engine, and then all the rest
def game_engine(sat_m, inter_m, steps, decay_rate, growth_rate=1):
    history = {
        "step": [],
        "active_behavior": [],
        "satisfaction": [],
        "unsatisfied_octants": [],
    }

    active_behavior = None
    if decay_params is None:
        decay_params = {
            "amplitude": 0.2,
            "elevation": 0.5,
            "angular_displacement": np.pi,
        }

    decay_generator = GenerateDecayMatrix()
    decay_matrix = decay_generator.individual_decay_sin(
        n_motives=len(sat_m.columns), **decay_params
    )

    for step in range(steps):
        # Get current satisfaction levels
        satisfaction_levels = sat_m.loc["satisfaction"]
        unsatisfied_octants = satisfaction_levels[
            satisfaction_levels < 0
        ].index.tolist()

        # Switch behavior of current one is satisfied (reached 1) or none is active
        if active_behavior is None or satisfaction_levels[active_behavior] >= 1:
            active_behavior = select_unsatisfied_behavior(
                satisfaction_levels, unsatisfied_octants
            )

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
                decay_rate = decay_matrix.loc["decay_rate", octant]
                sat_m.loc["satisfaction", octant] -= decay_rate
        # Clip all satisfaction values to be within the range [-1, 1]
        sat_m.loc["satisfaction"] = np.clip(sat_m.loc["satisfaction"], -1, 1)

        # Record history (after all updates)
        history["step"].append(step)
        history["active_behavior"].append(active_behavior)
        history["satisfaction"].append(sat_m.copy())
        history["unsatisfied_octants"].append(unsatisfied_octants)

    return history


def select_unsatisfied_behavior(satisfaction_levels, unsatisfied_octants):
    if not unsatisfied_octants:
        return None
    probs = -satisfaction_levels[unsatisfied_octants]
    probs = probs / probs.sum()  # Normalize to sum to 1
    return random.choices(unsatisfied_octants, weights=probs, k=1)[0]

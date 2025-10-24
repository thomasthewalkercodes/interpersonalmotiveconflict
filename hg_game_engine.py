from hg_lambda_calc import generate_interaction_matrix
from hg_lambda_calc import get_lambda
import numpy as np
import pandas as pd
import random

steps = 50
decay_rate = 0.1


def select_behavior(satisfaction_levels, unsatisfied_octants):
    if not unsatisfied_octants:
        return None
    probs = -satisfaction_levels[unsatisfied_octants]
    probs = probs / probs.sum()  # Normalize to sum to 1
    return random.choices(unsatisfied_octants, weights=probs, k=1)[0]


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


# example usage
if __name__ == "__main__":
    print("EXAMPLE USAGE OF GAME ENGINE WITH INTERACTION AND SATISFACTION MATRICES")
    inter_m = generate_interaction_matrix(n_motives=8, mean=0, sd=0.3)
    sat_m = generate_satisfaction_matrix(n_motives=8, mean=0.2, sd=0.3)
    print("Generated Interaction Matrix:")
    print(inter_m.round(3))
    print("\nGenerated Satisfaction Matrix:")
    print(sat_m.round(3))
    decay_rate = get_lambda(inter_m, decay_lambda=None)
    print(f"\nUsing Decay Rate (Lambda): {decay_rate['decay_lambda']:.4f}")
    game_engine_history = game_engine(
        sat_m, inter_m, steps=100, decay_rate=decay_rate["decay_lambda"], growth_rate=1
    )
    print("Game engine simulation completed.")
    # Print final satisfaction levels
    final_satisfaction = game_engine_history["satisfaction"][-1]
    print("\nFinal Satisfaction Levels:")
    print(final_satisfaction.round(3))
    # print active behavior history
    print("\nActive Behavior History:")
    for i, active in enumerate(game_engine_history["active_behavior"]):
        print(f"Step {i}: Active Behavior: {active}")

    # understand bleeding and saturation

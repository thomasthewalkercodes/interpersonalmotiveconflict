import random
import pandas as pd

steps = 50
decay_rate = 0.1

def game_engine(satisfaction_matrix, influence_matrix, steps, decay_rate, growth_rate =1):
    history = {
        "step": [],
        "active_behavior": [],
        "satisfaction": [],
        "unsatisfied_octants": [],
    }

    active_behavior = None

    for step in range(steps):
        unsatisfied_octants = satisfaction_matrix.columns[
            satisfaction_matrix.loc["satisfaction"] < 0
        ].tolist()
        # determine active behavior
        if active_behavior is None:
            if len(unsatisfied_octants) == 0:
                active_behavior = None  # checks if something is active
            elif len(unsatisfied_octants) > 0:
                # selection based on unsatisfaction level
                probs = -satisfaction_matrix.loc["satisfaction", unsatisfied_octants]
                probs = probs / probs.sum()  # Normalize to sum to 1
                active_behavior = random.choices(unsatisfied_octants, weights=probs, k=1)[0]

        else:
            if (
                active_behavior is not None
                and satisfaction_matrix.loc["satisfaction", active_behavior] >= 1
            ):
                active_behavior = None
                unsatisfied_octants = satisfaction_matrix.columns[
                    satisfaction_matrix.loc["satisfaction"] < 0
                ].tolist()
                if len(unsatisfied_octants) == 0:
                    active_behavior = None  # checks if something is active
                elif len(unsatisfied_octants) > 0:
                    # selection based on unsatisfaction level
                    probs = -satisfaction_matrix.loc["satisfaction", unsatisfied_octants]
                    probs = probs / probs.sum()  # Normalize to sum to 1
                    active_behavior = random.choices(
                        unsatisfied_octants, weights=probs, k=1
                    )[0]

        satisfaction_matrix = (
            satisfaction_matrix - decay_rate
        )  

        if active_behavior is not None:
            for octant in satisfaction_matrix.columns:
                if octant != active_behavior:
                    influence = influence_matrix.loc[active_behavior, octant]
                    satisfaction_matrix.loc["satisfaction", octant] += influence
            satisfaction_matrix.loc["satisfaction", active_behavior] += growth_rate  # growth rate

        # Record history
        history["step"].append(step)
        history["active_behavior"].append(active_behavior)
        history["satisfaction"].append(
            satisfaction_matrix.loc["satisfaction"].copy
"""
My interpretation of how this motive conflict model could be implemented in code.
"""

import numpy as np
import random
import pandas as pd
from cosinus_corr_matrix import create_circumplex_correlation_matrix


steps = 150
influence_multiplier = 0.02
growth_rate = 1


influence_of_octants = ["LM", "NO", "AP", "BC", "DE", "FG", "HI", "JK"]

df_influence_of_octants = pd.DataFrame(
    np.zeros((8, 8)), index=influence_of_octants, columns=influence_of_octants
)
lmno = df_influence_of_octants.loc["LM", "NO"] = 0.2
lmap = df_influence_of_octants.loc["LM", "AP"] = 0.2
lmbc = df_influence_of_octants.loc["LM", "BC"] = -0.4
lmde = df_influence_of_octants.loc["LM", "DE"] = -0.4
lmfg = df_influence_of_octants.loc["LM", "FG"] = -0.3
lmhi = df_influence_of_octants.loc["LM", "HI"] = -0.1
lmjk = df_influence_of_octants.loc["LM", "JK"] = 0.1

noap = df_influence_of_octants.loc["NO", "AP"] = 0.5
nobc = df_influence_of_octants.loc["NO", "BC"] = 0.1
node = df_influence_of_octants.loc["NO", "DE"] = 0.3
nofg = df_influence_of_octants.loc["NO", "FG"] = 0.4
nohi = df_influence_of_octants.loc["NO", "HI"] = 0.1
nojk = df_influence_of_octants.loc["NO", "JK"] = 0.1

apbc = df_influence_of_octants.loc["AP", "BC"] = -0.3
apde = df_influence_of_octants.loc["AP", "DE"] = 0.3
apfg = df_influence_of_octants.loc["AP", "FG"] = 0.1
aphi = df_influence_of_octants.loc["AP", "HI"] = -0.1
apjk = df_influence_of_octants.loc["AP", "JK"] = 0.1

bcde = df_influence_of_octants.loc["BC", "DE"] = 0.1
bcfg = df_influence_of_octants.loc["BC", "FG"] = 0.1
bchi = df_influence_of_octants.loc["BC", "HI"] = 0.1
bcjk = df_influence_of_octants.loc["BC", "JK"] = -0.1

defg = df_influence_of_octants.loc["DE", "FG"] = -0.3
dehi = df_influence_of_octants.loc["DE", "HI"] = 0.1
dejk = df_influence_of_octants.loc["DE", "JK"] = -0.2

fghi = df_influence_of_octants.loc["FG", "HI"] = 0.1
fgjk = df_influence_of_octants.loc["FG", "JK"] = 0.3

hijk = df_influence_of_octants.loc["HI", "JK"] = -0.3
# print(df_influence_of_octants)

df_influence_of_octants = df_influence_of_octants + df_influence_of_octants.T
print(df_influence_of_octants)

df_influence_of_octants = create_circumplex_correlation_matrix(
    n_motives=8,
    elevation=-0.01,
    amplitude=0.3,
    displacement=0,  # displacement in degrees, how much the curve shifts towards a different point
).round(3)

print(df_influence_of_octants)

octants = ["LM", "NO", "AP", "BC", "DE", "FG", "HI", "JK"]
df_satisfaction = pd.DataFrame(
    np.clip(np.random.normal(0.4, 0.4, (1, 8)), -1, 1),
    index=["satisfaction"],
    columns=octants,
)
print(df_satisfaction)

mean_influence = df_influence_of_octants.mean().mean()
print("Mean influence:", mean_influence)

history = {
    "step": [],
    "active_behavior": [],
    "satisfaction": [],
    "unsatisfied_octants": [],
}

active_behavior = None

for step in range(steps):
    unsatisfied_octants = df_satisfaction.columns[
        df_satisfaction.loc["satisfaction"] < 0
    ].tolist()
    # determine active behavior
    if active_behavior is None:
        if len(unsatisfied_octants) == 0:
            active_behavior = None  # checks if something is active
        elif len(unsatisfied_octants) > 0:
            # selection based on unsatisfaction level
            probs = -df_satisfaction.loc["satisfaction", unsatisfied_octants]
            probs = probs / probs.sum()  # Normalize to sum to 1
            active_behavior = random.choices(unsatisfied_octants, weights=probs, k=1)[0]

    else:
        if (
            active_behavior is not None
            and df_satisfaction.loc["satisfaction", active_behavior] >= 1
        ):
            active_behavior = None
            unsatisfied_octants = df_satisfaction.columns[
                df_satisfaction.loc["satisfaction"] < 0
            ].tolist()
            if len(unsatisfied_octants) == 0:
                active_behavior = None  # checks if something is active
            elif len(unsatisfied_octants) > 0:
                # selection based on unsatisfaction level
                probs = -df_satisfaction.loc["satisfaction", unsatisfied_octants]
                probs = probs / probs.sum()  # Normalize to sum to 1
                active_behavior = random.choices(
                    unsatisfied_octants, weights=probs, k=1
                )[0]

    df_satisfaction = (
        df_satisfaction - (1 - mean_influence) * influence_multiplier
    )  # assumption

    if active_behavior is not None:
        for octant in octants:
            if octant != active_behavior:
                influence = df_influence_of_octants.loc[active_behavior, octant]
                df_satisfaction.loc["satisfaction", octant] += influence
        df_satisfaction.loc["satisfaction", active_behavior] += growth_rate

    df_satisfaction = df_satisfaction.clip(lower=-1, upper=1)
    # Record current state in history
    history["step"].append(step)
    history["active_behavior"].append(active_behavior)
    history["satisfaction"].append(df_satisfaction.loc["satisfaction"].copy())
    history["unsatisfied_octants"].append(unsatisfied_octants)
    # convert to data frame
    df_history = pd.DataFrame(
        {
            "step": history["step"],
            "active_behavior": history["active_behavior"],
            "unsatisfied_octants": history["unsatisfied_octants"],
        }
    )

df_satisfaction_history = pd.DataFrame(history["satisfaction"])
df_satisfaction_history.index = history["step"]

print("\nSatisfaction History:")
print(df_satisfaction_history.round(3))

print(f"\nFinal satisfaction levels:")
print(df_satisfaction.loc["satisfaction"].round(3))

print(f"\nActive behavior history:")
for i, active in enumerate(history["active_behavior"]):  # Show first 20 steps
    print(f"Step {i}: {active}")

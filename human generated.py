"""
My interpretation of how this motive conflict model could be implemented in code.
"""

import numpy as np
import random
import pandas as pd

steps = 50
influence_multiplier = 0.15
network_impact = 0.2


influence_of_octants = ["LM", "NO", "AP", "BC", "DE", "FG", "HI", "JK"]

df_influence_of_octants = pd.DataFrame(
    np.zeros((8, 8)), index=influence_of_octants, columns=influence_of_octants
)
lmno = df_influence_of_octants.loc["LM", "NO"] = 0.5
lmap = df_influence_of_octants.loc["LM", "AP"] = 0.5
lmbc = df_influence_of_octants.loc["LM", "BC"] = -0.3
lmde = df_influence_of_octants.loc["LM", "DE"] = -0.3
lmfg = df_influence_of_octants.loc["LM", "FG"] = -0.1
lmhi = df_influence_of_octants.loc["LM", "HI"] = -0.1
lmjk = df_influence_of_octants.loc["LM", "JK"] = 0.1
noap = df_influence_of_octants.loc["NO", "AP"] = 0.5
nobc = df_influence_of_octants.loc["NO", "BC"] = -0.3
node = df_influence_of_octants.loc["NO", "DE"] = -0.3
nofg = df_influence_of_octants.loc["NO", "FG"] = -0.1
nohi = df_influence_of_octants.loc["NO", "HI"] = -0.1
nojk = df_influence_of_octants.loc["NO", "JK"] = 0.1
apbc = df_influence_of_octants.loc["AP", "BC"] = -0.3
apde = df_influence_of_octants.loc["AP", "DE"] = -0.3
apfg = df_influence_of_octants.loc["AP", "FG"] = -0.1
aphi = df_influence_of_octants.loc["AP", "HI"] = -0.1
apjk = df_influence_of_octants.loc["AP", "JK"] = 0.1
bcde = df_influence_of_octants.loc["BC", "DE"] = 0.5
bcfg = df_influence_of_octants.loc["BC", "FG"] = -0.3
bchi = df_influence_of_octants.loc["BC", "HI"] = -0.3
bcjk = df_influence_of_octants.loc["BC", "JK"] = -0.1
defg = df_influence_of_octants.loc["DE", "FG"] = -0.3
dehi = df_influence_of_octants.loc["DE", "HI"] = -0.3
dejk = df_influence_of_octants.loc["DE", "JK"] = -0.1
fghi = df_influence_of_octants.loc["FG", "HI"] = 0.5
fgjk = df_influence_of_octants.loc["FG", "JK"] = -0.3
hijk = df_influence_of_octants.loc["HI", "JK"] = -0.3
# print(df_influence_of_octants)

octants = ["LM", "NO", "AP", "BC", "DE", "FG", "HI", "JK"]
df_satisfaction = pd.DataFrame(np.ones((1, 8)), index=["satisfaction"], columns=octants)
# print(df_octants)
mean_influence = df_influence_of_octants.mean().mean()
print("Mean influence:", mean_influence)

for step in range(1, steps):
    unsatisfied_octants = df_satisfaction.columns[df_satisfaction.loc["satisfaction"] < 0].tolist() 
    if active_behavior is None:
        if len(unsatisfied_octants) == 0:
            active_behavior = None #checks if something is active
        elif len(unsatisfied_octants) > 1:
            probs = -df_satisfaction.loc["satisfaction", unsatisfied_octants]
            probs = probs / probs.sum()  # Normalize to sum to 1
            active_behavior = random.choices(unsatisfied_octants, weights=probs, k=1)[0]
            
    else:
        if active_behavior < 1:
        else: 
            active_behavior = None

    df_satisfaction = df_satisfaction - (1-mean_influence) * influence_multiplier #assumption
    if active_behavior is not None:
        for i in 1:7:
        df_satisfaction[i] = df_satisfaction[i] + (1 - df_influence_of_octants.loc[active_behavior, i]) * network_impact #assumption
        
    df_satisfaction = df_satisfaction.clip(lower=-1, upper=1)
    
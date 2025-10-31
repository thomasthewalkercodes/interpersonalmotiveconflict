# This might get used at a later point since growth is always +1
# but I guess it doesnt hurt
import numpy as np
import pandas as pd


class GenerateGrowthMatrix:
    def one_growth(n_motives=8):
        growth_rate_values = np.ones(n_motives)

        return pd.DataFrame(
            [growth_rate_values],
            columns=[f"motive_{i+1}" for i in range(n_motives)],
            index=["growth_rate"],
        )

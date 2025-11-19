import sys
import os

from pathlib import Path
from datetime import datetime
import pandas as pd

from hg_full_game_engine import game_engine
from interaction_matrix import GenerateInteractionMatrix
from satisfaction_matrix import SatisfactonMatrixGenerator
from decay_matrix import GenerateDecayMatrix

steps = 50


def run_single_simulation(steps):
    sat_m = SatisfactonMatrixGenerator().normal_distribution_sat_matrix(
        n_motives=8, mean=0, sd=0.5
    )
    inter_m = GenerateInteractionMatrix().circumplex_int_matrix(
        n_motives=8, start_motive=1, amplitude=0.3, elevation=0.1
    )
    decay_m = GenerateDecayMatrix().individual_decay_sin(
        start_motive=1, amplitude=0.1, elevation=0.2
    )
    growth_rate = 1  # Fixed growth rate
    game_history = game_engine(sat_m, inter_m, steps, decay_m, growth_rate)
    return game_history, inter_m, sat_m, decay_m, growth_rate


def run_batch_simulations(n_simulations, steps, base_output_dir="batch_output"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = Path(f"{base_output_dir}_{timestamp}")
    batch_dir.mkdir(parents=True, exist_ok=True)

    all_histories = []

    for sim_idx in range(n_simulations):
        history, inter_m, sat_m, decay_m, growth_rate = run_single_simulation(steps)
        inter_m.to_csv(batch_dir / f"interaction_matrix_sim_{sim_idx+1}.csv")
        sat_m.to_csv(batch_dir / f"satisfaction_matrix_sim_{sim_idx+1}.csv")
        decay_m.to_csv(batch_dir / f"decay_matrix_sim_{sim_idx+1}.csv")
        # Save growth_rate as a simple CSV with scalar value
        pd.DataFrame([{"growth_rate": growth_rate}]).to_csv(
            batch_dir / f"growth_rate_sim_{sim_idx+1}.csv", index=False
        )
        all_histories.append(history)
    print(f"Batch simulations saved to: {batch_dir}")
    return all_histories


histories = run_batch_simulations(
    n_simulations=10, steps=50, base_output_dir="my_batch"
)

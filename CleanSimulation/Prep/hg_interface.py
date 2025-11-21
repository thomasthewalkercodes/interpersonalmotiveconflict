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
        n_motives=8, mean=0.2, sd=0.2
    )
    """inter_m = GenerateInteractionMatrix().circumplex_int_matrix(
        n_motives=8, start_motive=1, amplitude=0.2, elevation=0.1
    )"""
    inter_m = GenerateInteractionMatrix().borderline_int_matrix(
        n_motives=8,
        start_motive=1,
        amplitude_dict={3: 0.3, 7: 0.3},  # Custom amplitudes for motives 3 and 6
        elevation_dict={3: 0.1, 7: 0.1},  # Custom elevations for motives 3 and 6
        base_amplitude=0.2,
        base_elevation=0.1,
    )
    """decay_m = GenerateDecayMatrix().individual_decay_sin(
        start_motive=3, amplitude=0.01, elevation=0.02
    )"""
    decay_m = GenerateDecayMatrix().matrix_specific_decay(inter_m, decay_lambda=None)

    growth_rate = 1  # Fixed growth rate
    game_history = game_engine(sat_m, inter_m, steps, decay_m, growth_rate)
    return game_history, inter_m, sat_m, decay_m, growth_rate


def run_batch_simulations(n_simulations, steps, base_output_dir="batch_output"):
    base_data_dir = Path(__file__).resolve().parents[1] / "data"
    base_data_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = base_data_dir / f"{base_output_dir}_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    all_histories = []

    for sim_idx in range(n_simulations):
        # Create subfolder for this simulation
        sim_folder = batch_dir / f"simulation_{sim_idx+1:03d}"
        sim_folder.mkdir(exist_ok=True)

        # Run simulation
        history, inter_m, sat_m, decay_m, growth_rate = run_single_simulation(steps)

        # Save input matrices
        inter_m.to_csv(sim_folder / "interaction_matrix.csv")
        sat_m.to_csv(sim_folder / "initial_satisfaction.csv")
        decay_m.to_csv(sim_folder / "decay_matrix.csv")
        pd.DataFrame([{"growth_rate": growth_rate}]).to_csv(
            sim_folder / "growth_rate.csv", index=False
        )

        # Save behavior sequence
        behavior_df = pd.DataFrame(
            {"step": history["step"], "active_behavior": history["active_behavior"]}
        )
        behavior_df.to_csv(sim_folder / "behavior_sequence.csv", index=False)

        # Save satisfaction timeseries (wide format)
        sat_timeseries = []
        for step_idx, sat_df in enumerate(history["satisfaction"]):
            row = {"step": history["step"][step_idx]}
            for col in sat_df.columns:
                row[col] = sat_df.loc["satisfaction", col]
            sat_timeseries.append(row)

        sat_timeseries_df = pd.DataFrame(sat_timeseries)
        sat_timeseries_df.to_csv(
            sim_folder / "satisfaction_timeseries.csv", index=False
        )

        all_histories.append(history)
        print(f"[OK] Saved simulation {sim_idx+1}/{n_simulations} to {sim_folder.name}")

    print(f"\n[OK] All {n_simulations} simulations saved to: {batch_dir}")
    return all_histories


histories = run_batch_simulations(
    n_simulations=1, steps=500, base_output_dir="my_batch"
)

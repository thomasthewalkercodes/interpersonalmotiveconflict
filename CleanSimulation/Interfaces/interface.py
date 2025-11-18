"""
Simple interface for running multiple simulations and saving results to CSV.
All parameters are saved for replication.
"""

import sys
import os
from datetime import datetime
import pandas as pd

# Add parent directory to path to import from Prep
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Prep'))

from hg_full_game_engine import game_engine
from interaction_matrix import GenerateInteractionMatrix
from satisfaction_matrix import SatisfactonMatrixGenerator
from decay_matrix import GenerateDecayMatrix


class SimulationRunner:
    """Runs multiple simulations and saves results to CSV files."""

    def __init__(self, results_dir="../Results"):
        """Initialize the simulation runner.

        Args:
            results_dir: Directory to save results (default: ../Results)
        """
        self.results_dir = results_dir
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)

    def run_single_simulation(self, params):
        """Run a single simulation with given parameters.

        Args:
            params: Dictionary containing simulation parameters

        Returns:
            history: Simulation history from game_engine
        """
        # Generate initial satisfaction matrix
        sat_gen = SatisfactonMatrixGenerator()
        sat_m = sat_gen.normal_distribution_sat_matrix(
            n_motives=params['n_motives'],
            mean=params['sat_mean'],
            sd=params['sat_sd']
        )

        # Generate interaction matrix
        inter_gen = GenerateInteractionMatrix()
        if params['interaction_type'] == 'circumplex':
            inter_m = inter_gen.circumplex_int_matrix(
                n_motives=params['n_motives'],
                start_motive=params['inter_start_motive'],
                amplitude=params['inter_amplitude'],
                elevation=params['inter_elevation']
            )
        else:  # normal distribution
            inter_m = inter_gen.normal_distribution_int_matrix(
                n_motives=params['n_motives'],
                mean=params['inter_mean'],
                sd=params['inter_sd']
            )

        # Generate decay matrix
        decay_gen = GenerateDecayMatrix()
        if params['decay_type'] == 'individual_sin':
            decay_m = decay_gen.individual_decay_sin(
                n_motives=params['n_motives'],
                start_motive=params['decay_start_motive'],
                amplitude=params['decay_amplitude'],
                elevation=params['decay_elevation']
            )
        else:  # matrix-specific
            decay_m = decay_gen.matrix_specific_decay(
                inter_m.values,
                decay_lambda=params.get('decay_lambda')
            )

        # Run the simulation
        history = game_engine(
            sat_m=sat_m.copy(),
            inter_m=inter_m,
            steps=params['steps'],
            decay_rate=decay_m,
            growth_rate=params['growth_rate']
        )

        return history, sat_m, inter_m, decay_m

    def save_simulation_results(self, sim_id, params, history, sat_m, inter_m, decay_m):
        """Save all simulation results to CSV files.

        Args:
            sim_id: Unique simulation identifier (timestamp)
            params: Simulation parameters
            history: Simulation history
            sat_m: Initial satisfaction matrix
            inter_m: Interaction matrix
            decay_m: Decay matrix
        """
        # Save input parameters
        params_df = pd.DataFrame([params])
        params_df.to_csv(
            f"{self.results_dir}/input_analysis_{sim_id}.csv",
            index=False
        )

        # Save initial satisfaction
        sat_m.to_csv(
            f"{self.results_dir}/initial_satisfaction_{sim_id}.csv"
        )

        # Save interaction matrix
        inter_m.to_csv(
            f"{self.results_dir}/interaction_matrix_{sim_id}.csv"
        )

        # Save decay matrix
        if isinstance(decay_m, pd.DataFrame):
            decay_m.to_csv(
                f"{self.results_dir}/decay_matrix_{sim_id}.csv"
            )
        else:
            # If decay_m is a set (from matrix_specific_decay)
            pd.DataFrame([{'decay_rate': list(decay_m)[0]}]).to_csv(
                f"{self.results_dir}/decay_matrix_{sim_id}.csv",
                index=False
            )

        # Save behavior sequence
        behavior_df = pd.DataFrame({
            'step': history['step'],
            'active_behavior': history['active_behavior']
        })
        behavior_df.to_csv(
            f"{self.results_dir}/behavior_sequence_{sim_id}.csv",
            index=False
        )

        # Save satisfaction timeseries
        # Flatten satisfaction history into a wide-format dataframe
        sat_timeseries = []
        for step_idx, sat_df in enumerate(history['satisfaction']):
            row = {'step': history['step'][step_idx]}
            for col in sat_df.columns:
                row[col] = sat_df.loc['satisfaction', col]
            sat_timeseries.append(row)

        sat_timeseries_df = pd.DataFrame(sat_timeseries)
        sat_timeseries_df.to_csv(
            f"{self.results_dir}/satisfaction_timeseries_{sim_id}.csv",
            index=False
        )

        print(f"[OK] Saved simulation {sim_id}")

    def run_batch_simulations(self, n_simulations, base_params):
        """Run multiple simulations with the same parameters.

        Args:
            n_simulations: Number of simulations to run
            base_params: Base parameters for all simulations

        Returns:
            List of simulation IDs
        """
        sim_ids = []

        print(f"\nRunning {n_simulations} simulations...")
        print(f"Parameters: {base_params}\n")

        for i in range(n_simulations):
            # Generate unique ID for this simulation
            sim_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{i:03d}"

            # Run simulation
            history, sat_m, inter_m, decay_m = self.run_single_simulation(base_params)

            # Save results
            self.save_simulation_results(sim_id, base_params, history, sat_m, inter_m, decay_m)

            sim_ids.append(sim_id)
            print(f"Completed simulation {i+1}/{n_simulations}")

        print(f"\n[OK] All simulations complete! Results saved to {self.results_dir}/")
        return sim_ids


def get_default_params():
    """Get default simulation parameters."""
    return {
        # Simulation settings
        'n_motives': 8,
        'steps': 100,
        'growth_rate': 1.0,

        # Initial satisfaction settings
        'sat_mean': 0.2,
        'sat_sd': 0.3,

        # Interaction matrix settings
        'interaction_type': 'circumplex',  # 'circumplex' or 'normal'
        'inter_start_motive': 1,  # for circumplex
        'inter_amplitude': 0.3,   # for circumplex
        'inter_elevation': 0.1,   # for circumplex
        'inter_mean': 0.0,        # for normal distribution
        'inter_sd': 0.2,          # for normal distribution

        # Decay settings
        'decay_type': 'individual_sin',  # 'individual_sin' or 'matrix_specific'
        'decay_start_motive': 1,  # for individual_sin
        'decay_amplitude': 0.1,   # for individual_sin
        'decay_elevation': 0.2,   # for individual_sin
        'decay_lambda': None,     # for matrix_specific (None = auto-calculate)
    }


if __name__ == "__main__":
    # Example usage
    runner = SimulationRunner(results_dir="../../Results")

    # Get default parameters
    params = get_default_params()

    # Customize parameters if needed
    params['steps'] = 100
    params['n_motives'] = 8
    params['interaction_type'] = 'circumplex'
    params['decay_type'] = 'individual_sin'

    # Run multiple simulations
    sim_ids = runner.run_batch_simulations(
        n_simulations=3,
        base_params=params
    )

    print(f"\nSimulation IDs: {sim_ids}")

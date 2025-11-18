"""
Simple test script for hg_full_game_engine
Generates input matrices, runs simulation, and saves analysis to CSV in Results folder
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import your existing modules (make sure they're in the same directory)
from interaction_matrix import GenerateInteractionMatrix
from satisfaction_matrix import SatisfactonMatrixGenerator
from decay_matrix import GenerateDecayMatrix


def analyze_matrix(matrix, name):
    """Calculate statistics for a matrix"""
    values = matrix.values.flatten()
    # Remove diagonal zeros for interaction matrix
    if name == "Influence":
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        values = matrix.values[mask]

    return {
        f"{name}_mean": np.mean(values),
        f"{name}_sd": np.std(values),
        f"{name}_min": np.min(values),
        f"{name}_max": np.max(values),
        f"{name}_median": np.median(values),
    }


def analyze_behavior_sequence(history):
    """Analyze the behavior sequence generated"""
    behaviors = [b for b in history["active_behavior"] if b is not None]

    if not behaviors:
        return {
            "total_steps": len(history["step"]),
            "unique_behaviors": 0,
            "most_common_behavior": "None",
            "behavior_switches": 0,
            "none_count": len(history["active_behavior"]),
        }

    # Count behavior switches
    switches = sum(
        1
        for i in range(1, len(history["active_behavior"]))
        if history["active_behavior"][i] != history["active_behavior"][i - 1]
    )

    # Get unique behaviors and their counts
    from collections import Counter

    behavior_counts = Counter(behaviors)
    most_common = behavior_counts.most_common(1)[0]

    return {
        "total_steps": len(history["step"]),
        "unique_behaviors": len(set(behaviors)),
        "most_common_behavior": most_common[0],
        "most_common_count": most_common[1],
        "behavior_switches": switches,
        "none_count": history["active_behavior"].count(None),
    }


def run_test_simulation(n_motives=8, steps=100, seed=42):
    """
    Run a simple test simulation and return all data

    Parameters:
    -----------
    n_motives : int
        Number of motives (default 8)
    steps : int
        Number of simulation steps (default 100)
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    dict : All simulation data and analysis
    """
    np.random.seed(seed)

    print(f"Generating matrices for {n_motives} motives...")

    # Generate interaction matrix
    inter_m = GenerateInteractionMatrix.normal_distribution_int_matrix(
        n_motives=n_motives, mean=0.0, sd=0.2
    )

    # Generate initial satisfaction
    sat_m = SatisfactonMatrixGenerator.normal_distribution_sat_matrix(
        n_motives=n_motives, mean=0.2, sd=0.3
    )
    sat_m_initial = sat_m.copy()

    # Generate decay parameters
    decay_params = {"amplitude": 0.1, "elevation": 0.2, "start_motive": 1}

    decay_matrix = GenerateDecayMatrix.individual_decay_sin(
        n_motives=n_motives, **decay_params
    )

    print(f"Running simulation for {steps} steps...")

    # Run simulation (using the fixed version)
    history = game_engine_fixed(
        sat_m=sat_m,
        inter_m=inter_m,
        steps=steps,
        decay_params=decay_params,
        growth_rate=1,
    )

    print("Analyzing results...")

    # Analyze inputs
    inter_stats = analyze_matrix(inter_m, "Influence")
    sat_stats = analyze_matrix(sat_m_initial, "Starting_Satisfaction")
    decay_stats = analyze_matrix(decay_matrix, "Decay")

    # Analyze behavior sequence
    behavior_stats = analyze_behavior_sequence(history)

    return {
        "inter_m": inter_m,
        "sat_m_initial": sat_m_initial,
        "decay_matrix": decay_matrix,
        "history": history,
        "stats": {**inter_stats, **sat_stats, **decay_stats, **behavior_stats},
    }


def game_engine_fixed(sat_m, inter_m, steps, decay_params, growth_rate=1):
    """
    Fixed version of game_engine that properly handles decay_params
    """
    history = {
        "step": [],
        "active_behavior": [],
        "satisfaction": [],
        "unsatisfied_octants": [],
    }

    active_behavior = None

    # Generate decay matrix
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

        # Switch behavior if current one is satisfied (reached 1) or none is active
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
    """Select behavior based on dissatisfaction levels"""
    import random

    if not unsatisfied_octants:
        return None
    probs = -satisfaction_levels[unsatisfied_octants]
    probs = probs / probs.sum()  # Normalize to sum to 1
    return random.choices(unsatisfied_octants, weights=probs, k=1)[0]


def save_results(results, output_dir="Results"):
    """Save all results to CSV files in the Results folder"""

    # Create Results directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\nSaving results to {output_dir}/...")

    # 1. Save input analysis (the main summary CSV)
    stats_df = pd.DataFrame([results["stats"]])
    stats_file = output_path / f"input_analysis_{timestamp}.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"[OK] Saved: {stats_file}")

    # 2. Save interaction matrix
    inter_file = output_path / f"interaction_matrix_{timestamp}.csv"
    results["inter_m"].to_csv(inter_file)
    print(f"[OK] Saved: {inter_file}")

    # 3. Save initial satisfaction
    sat_file = output_path / f"initial_satisfaction_{timestamp}.csv"
    results["sat_m_initial"].to_csv(sat_file)
    print(f"[OK] Saved: {sat_file}")

    # 4. Save decay matrix
    decay_file = output_path / f"decay_matrix_{timestamp}.csv"
    results["decay_matrix"].to_csv(decay_file)
    print(f"[OK] Saved: {decay_file}")

    # 5. Save behavior sequence
    behavior_df = pd.DataFrame(
        {
            "step": results["history"]["step"],
            "active_behavior": results["history"]["active_behavior"],
        }
    )
    behavior_file = output_path / f"behavior_sequence_{timestamp}.csv"
    behavior_df.to_csv(behavior_file, index=False)
    print(f"[OK] Saved: {behavior_file}")

    # 6. Save satisfaction over time (long format for easy plotting)
    satisfaction_data = []
    for step_idx, step in enumerate(results["history"]["step"]):
        sat_df = results["history"]["satisfaction"][step_idx]
        for motive in sat_df.columns:
            satisfaction_data.append(
                {
                    "step": step,
                    "motive": motive,
                    "satisfaction": sat_df.loc["satisfaction", motive],
                }
            )

    sat_time_df = pd.DataFrame(satisfaction_data)
    sat_time_file = output_path / f"satisfaction_timeseries_{timestamp}.csv"
    sat_time_df.to_csv(sat_time_file, index=False)
    print(f"[OK] Saved: {sat_time_file}")

    print(f"\n[OK] All results saved to {output_dir}/")
    return output_path


def main():
    """Main execution function"""
    print("=" * 60)
    print("SIMPLE TEST RUN - HG_FULL_GAME_ENGINE")
    print("=" * 60)

    # Run test simulation
    results = run_test_simulation(n_motives=8, steps=100, seed=42)

    # Save results
    output_dir = save_results(results)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nInput Parameters:")
    print(f"  Influence Matrix Mean: {results['stats']['Influence_mean']:.3f}")
    print(f"  Influence Matrix SD: {results['stats']['Influence_sd']:.3f}")
    print(
        f"  Starting Satisfaction Mean: {results['stats']['Starting_Satisfaction_mean']:.3f}"
    )
    print(
        f"  Starting Satisfaction SD: {results['stats']['Starting_Satisfaction_sd']:.3f}"
    )
    print(f"  Decay Mean: {results['stats']['Decay_mean']:.3f}")
    print(f"  Decay SD: {results['stats']['Decay_sd']:.3f}")

    print(f"\nBehavior Generation:")
    print(f"  Total Steps: {results['stats']['total_steps']}")
    print(f"  Unique Behaviors: {results['stats']['unique_behaviors']}")
    print(
        f"  Most Common: {results['stats']['most_common_behavior']} ({results['stats']['most_common_count']} times)"
    )
    print(f"  Behavior Switches: {results['stats']['behavior_switches']}")
    print(f"  No Behavior (None): {results['stats']['none_count']} steps")

    print("\n" + "=" * 60)
    print("[OK] TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

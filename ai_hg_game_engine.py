from hg_lambda_calc import generate_interaction_matrix
from hg_lambda_calc import get_lambda
from ai_visualizer import visualize_simulation
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
    print("=" * 70)
    print("MOTIVE GAME ENGINE - FULL SIMULATION WITH VISUALIZATION")
    print("=" * 70)

    # Set random seed for reproducibility
    np.random.seed(41)
    random.seed(41)

    # Simulation parameters
    n_motives = 8
    steps = 400
    growth_rate = 1

    print("\n1. Generating interaction matrix...")
    inter_m = generate_interaction_matrix(n_motives=n_motives, mean=0, sd=0.3)
    print("   Interaction Matrix:")
    print(inter_m.round(3))

    print("\n2. Generating initial satisfaction levels...")
    sat_m = generate_satisfaction_matrix(n_motives=n_motives, mean=0.2, sd=0.3)
    print("   Initial Satisfaction:")
    print(sat_m.round(3))

    print("\n3. Calculating decay rate...")
    lambda_info = get_lambda(inter_m, decay_lambda=None)
    decay_rate = lambda_info["decay_lambda"]
    print(f"   Using Decay Rate (Lambda): {decay_rate:.4f}")
    print(
        f"   Mean Interaction Strength: {lambda_info.get('mean_interaction_strength', 'N/A')}"
    )

    print("\n4. Running simulation...")
    sat_m_initial = sat_m.copy()  # Save initial state for analysis

    game_history = game_engine(
        sat_m, inter_m, steps=steps, decay_rate=decay_rate, growth_rate=growth_rate
    )

    print(f"   ✓ Simulation completed: {steps} steps")

    print("\n5. Final satisfaction levels:")
    final_satisfaction = game_history["satisfaction"][-1]
    print(final_satisfaction.round(3))

    print("\n6. Running comprehensive analysis and generating visualizations...")

    # Prepare parameters dictionary
    params = {
        "n_motives": n_motives,
        "steps": steps,
        "decay_rate": decay_rate,
        "growth_rate": growth_rate,
        "mean_interaction": lambda_info.get("mean_interaction_strength", 0),
        "calculation_method": lambda_info.get("calculation_method", "manual"),
    }

    # Run full analysis and visualization
    analyzer = visualize_simulation(
        history=game_history,
        inter_m=inter_m,
        sat_m_initial=sat_m_initial,
        params=params,
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nAll results saved to: {analyzer.output_dir}")
    print("\nGenerated files:")
    print("  📊 Visualizations:")
    print("     - satisfaction_over_time_[timestamp].png")
    print("     - active_behaviors_[timestamp].png")
    print("     - behavior_analysis_[timestamp].png")
    print("     - network_analysis_[timestamp].png")
    print("     - markov_chain_[timestamp].png")
    print("\n  📄 Data files:")
    print("     - parameters_[timestamp].csv")
    print("     - interaction_matrix_[timestamp].csv")
    print("     - initial_satisfaction_[timestamp].csv")
    print("     - metrics_[timestamp].json")
    print("     - metrics_summary_[timestamp].csv")
    print("     - behavior_sequence_[timestamp].csv")
    print("     - satisfaction_timeseries_[timestamp].csv")
    print("\n" + "=" * 70)

    # Print some key metrics
    print("\n📈 KEY METRICS SUMMARY:")
    print("-" * 70)

    # Get metrics from analyzer
    metrics = analyzer.calculate_all_metrics()

    print(f"\nINPUT CHARACTERISTICS:")
    print(f"  • Mean Congruence: {metrics['input']['mean_congruence']:.3f}")
    print(f"  • Mean Conflict: {metrics['input']['mean_conflict']:.3f}")
    print(
        f"  • Initial Mean Satisfaction: {metrics['input']['mean_initial_satisfaction']:.3f}"
    )
    print(f"  • Network Density: {metrics['input']['network']['density']:.3f}")

    print(f"\nOUTPUT CHARACTERISTICS:")
    print(
        f"  • Final Mean Satisfaction: {metrics['output']['mean_satisfaction_over_time']:.3f}"
    )
    print(
        f"  • Mean Behavior Duration: {metrics['output']['behavior_rhythm']['mean_duration']:.2f} steps"
    )
    print(
        f"  • Number of Behavior Switches: {metrics['output']['behavior_rhythm']['switch_count']}"
    )
    print(
        f"  • Mean Satisfied Duration: {metrics['output']['satisfaction_durations']['satisfied']['mean']:.2f} steps"
    )
    print(
        f"  • Mean Unsatisfied Duration: {metrics['output']['satisfaction_durations']['unsatisfied']['mean']:.2f} steps"
    )

    print("\n" + "=" * 70)
    print("You can now examine the generated plots and data files!")
    print("=" * 70)

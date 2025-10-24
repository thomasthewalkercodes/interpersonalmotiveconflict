"""
Optimized Batch Processing for Motive Game Engine
Uses external ai_pattern_analysis module for pattern detection.
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend BEFORE importing pyplot

from hg_lambda_calc import generate_interaction_matrix, get_lambda
from hg_game_engine import game_engine, generate_satisfaction_matrix
from ai_pattern_analysis import PatternDetector, StructuralPatternDetector
import numpy as np
import pandas as pd
import random
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
import json
import gc  # For garbage collection


class BatchSimulationProcessor:
    """Handles batch processing of simulations with optimized storage."""

    def __init__(self, base_output_dir="batch_simulations"):
        """Initialize batch processor."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(f"{base_output_dir}_{timestamp}")
        self.base_dir.mkdir(exist_ok=True)

        # Create subdirectories
        self.data_dir = self.base_dir / "data"
        self.plots_dir = self.base_dir / "plots"
        self.analysis_dir = self.base_dir / "analysis"

        self.data_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        self.analysis_dir.mkdir(exist_ok=True)

        # Storage for cross-simulation analysis
        self.all_behavior_sequences = []
        self.all_patterns = []
        self.simulation_metadata = []

        print(f"📁 Batch simulation directory created: {self.base_dir}")
        print(f"   ├── data/      (matrices & satisfaction timeseries)")
        print(f"   ├── plots/     (satisfaction_over_time plots)")
        print(f"   └── analysis/  (patterns & similarity analysis)")

    def run_single_simulation(
        self,
        seed,
        n_motives=8,
        steps=2000,
        growth_rate=1,
        inter_mean=0.2,
        inter_sd=0.3,
        sat_mean=0.2,
        sat_sd=0.3,
    ):
        """
        Run a single simulation with minimal storage.

        Args:
            seed: Random seed for reproducibility
            n_motives: Number of motives
            steps: Simulation steps
            growth_rate: Growth rate parameter
            inter_mean, inter_sd: Interaction matrix parameters
            sat_mean, sat_sd: Initial satisfaction parameters

        Returns:
            dict with results
        """
        # Set seed
        np.random.seed(seed)
        random.seed(seed)

        # Generate matrices
        inter_m = generate_interaction_matrix(
            n_motives=n_motives, mean=inter_mean, sd=inter_sd
        )
        sat_m = generate_satisfaction_matrix(
            n_motives=n_motives, mean=sat_mean, sd=sat_sd
        )
        sat_m_initial = sat_m.copy()

        # Get decay rate
        lambda_info = get_lambda(inter_m, decay_lambda=None)
        decay_rate = lambda_info["decay_lambda"]

        # Run simulation
        game_history = game_engine(
            sat_m, inter_m, steps=steps, decay_rate=decay_rate, growth_rate=growth_rate
        )

        # Extract satisfaction array
        satisfaction_array = self._extract_satisfaction_array(game_history, n_motives)
        behavior_sequence = game_history["active_behavior"]

        # Save data (skip plotting for now to debug)
        self._save_simulation_data(
            seed,
            inter_m,
            sat_m_initial,
            satisfaction_array,
            game_history["step"],
            behavior_sequence,
        )

        # Create satisfaction plot (with error handling)
        try:
            self._plot_satisfaction(
                seed, satisfaction_array, game_history["step"], list(inter_m.columns)
            )
        except Exception as e:
            print(f"      ⚠️ Plot failed for sim {seed}: {str(e)[:50]}...")

        # Pattern detection (STRUCTURAL - focuses on pattern structure, not specific motives)
        try:
            # Detect both exact and structural patterns
            exact_pattern = PatternDetector.find_longest_repeating_pattern(
                behavior_sequence,
                max_pattern_length=30,
                timeout_seconds=3,
                structural=False,  # Exact motive matching
            )

            structural_pattern = PatternDetector.find_longest_repeating_pattern(
                behavior_sequence,
                max_pattern_length=30,
                timeout_seconds=3,
                structural=True,  # Structure-based matching (A→B→A regardless of which motives)
            )

            pattern_info = structural_pattern  # Use structural for main analysis

        except Exception as e:
            print(f"      ⚠️ Pattern detection failed for sim {seed}: {str(e)[:50]}...")
            pattern_info = None
            structural_pattern = None

        # Store for cross-simulation analysis
        self.all_behavior_sequences.append(
            {"seed": seed, "sequence": behavior_sequence}
        )

        if pattern_info:
            pattern_info["seed"] = seed
            self.all_patterns.append(pattern_info)

        # Store metadata (include structural pattern info)
        self.simulation_metadata.append(
            {
                "seed": seed,
                "n_motives": n_motives,
                "steps": steps,
                "decay_rate": decay_rate,
                "growth_rate": growth_rate,
                "inter_mean": inter_mean,
                "inter_sd": inter_sd,
                "sat_mean": sat_mean,
                "sat_sd": sat_sd,
                "has_pattern": pattern_info is not None,
                "pattern_length": pattern_info["length"] if pattern_info else None,
                "pattern_occurrences": (
                    pattern_info["num_occurrences"] if pattern_info else None
                ),
                "pattern_structure": (
                    pattern_info.get("structure_string", None) if pattern_info else None
                ),
            }
        )

        # Clean up memory periodically
        if seed % 10 == 0:
            gc.collect()

        return {
            "seed": seed,
            "pattern": pattern_info,
            "behavior_sequence_length": len(
                [b for b in behavior_sequence if b is not None]
            ),
        }

    def _extract_satisfaction_array(self, history, n_motives):
        """Extract satisfaction history as numpy array."""
        n_steps = len(history["step"])
        sat_array = np.zeros((n_steps, n_motives))

        for i, sat_df in enumerate(history["satisfaction"]):
            sat_array[i, :] = sat_df.loc["satisfaction"].values

        return sat_array

    def _save_simulation_data(
        self, seed, inter_m, sat_m_initial, satisfaction_array, steps, behavior_sequence
    ):
        """Save minimal data for each simulation."""
        seed_dir = self.data_dir / f"sim_{seed:04d}"
        seed_dir.mkdir(exist_ok=True)

        # Save matrices
        inter_m.to_csv(seed_dir / f"interaction_matrix_{seed:04d}.csv")
        sat_m_initial.to_csv(seed_dir / f"initial_satisfaction_{seed:04d}.csv")

        # Save satisfaction timeseries
        sat_df = pd.DataFrame(satisfaction_array, columns=inter_m.columns)
        sat_df.insert(0, "step", steps)
        sat_df.to_csv(seed_dir / f"satisfaction_timeseries_{seed:04d}.csv", index=False)

        # Save behavior sequence
        behavior_df = pd.DataFrame(
            {"step": steps, "active_behavior": behavior_sequence}
        )
        behavior_df.to_csv(seed_dir / f"behavior_sequence_{seed:04d}.csv", index=False)

    def _plot_satisfaction(self, seed, satisfaction_array, steps, motive_names):
        """Create and save satisfaction over time plot with improved memory management."""
        try:
            fig, ax = plt.subplots(figsize=(14, 8))

            n_motives = len(motive_names)
            colors = plt.cm.tab10(np.linspace(0, 1, n_motives))

            for i, motive in enumerate(motive_names):
                ax.plot(
                    steps,
                    satisfaction_array[:, i],
                    label=motive,
                    color=colors[i],
                    linewidth=1.5,
                    alpha=0.8,
                )

            # Red line at y=0
            ax.axhline(
                y=0,
                color="red",
                linestyle="-",
                linewidth=2,
                alpha=0.7,
                label="Satisfaction Threshold (0)",
                zorder=0,
            )

            # Grey boundaries
            ax.axhline(y=1, color="grey", linestyle="--", linewidth=1, alpha=0.5)
            ax.axhline(y=-1, color="grey", linestyle="--", linewidth=1, alpha=0.5)

            ax.set_xlabel("Time Step", fontsize=12)
            ax.set_ylabel("Satisfaction Level", fontsize=12)
            ax.set_title(
                f"Motive Satisfaction Over Time - Simulation {seed:04d}",
                fontsize=14,
                fontweight="bold",
            )
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-1.1, 1.1)

            # Save with tight layout calculated first
            plt.tight_layout()

            # Save the figure
            output_path = self.plots_dir / f"satisfaction_over_time_{seed:04d}.png"
            fig.savefig(output_path, dpi=150, bbox_inches="tight", format="png")

            # Close figure and clear memory
            plt.close(fig)
            del fig, ax

        except Exception as e:
            print(f"⚠️  Warning: Could not create plot for simulation {seed}: {e}")
            # Continue processing even if plot fails
            if "fig" in locals():
                plt.close(fig)

    def analyze_patterns(self):
        """Analyze all detected patterns and save results."""
        if not self.all_patterns:
            print("⚠️  No repeating patterns detected in any simulation")
            print("    (Note: Only ACTIVE behaviors analyzed, None values skipped)")
            return

        # Create patterns dataframe with structural information
        patterns_data = []
        for p in self.all_patterns:
            pattern_str = "→".join([str(x) for x in p["pattern"]])
            structure_str = p.get("structure_string", "N/A")

            # Get examples if available
            examples = []
            if "all_examples" in p:
                for ex in p["all_examples"][:3]:  # First 3 examples
                    examples.append("→".join([str(x) for x in ex]))

            patterns_data.append(
                {
                    "seed": p["seed"],
                    "pattern": pattern_str,
                    "structure": structure_str,  # e.g., "A→B→A→C"
                    "pattern_length": p["length"],
                    "first_occurrence_step": p["first_occurrence"],
                    "num_occurrences": p["num_occurrences"],
                    "occurrence_positions": str(p["occurrence_positions"]),
                    "example_1": examples[0] if len(examples) > 0 else "",
                    "example_2": examples[1] if len(examples) > 1 else "",
                    "example_3": examples[2] if len(examples) > 2 else "",
                }
            )

        patterns_df = pd.DataFrame(patterns_data)
        patterns_df.to_csv(self.analysis_dir / "detected_patterns.csv", index=False)

        print(f"✓ Pattern analysis saved: {len(self.all_patterns)} patterns detected")
        print(f"  (Analyzed ACTIVE behaviors only, structural patterns identified)")
        print(f"  - Mean pattern length: {patterns_df['pattern_length'].mean():.2f}")
        print(f"  - Mean occurrences: {patterns_df['num_occurrences'].mean():.2f}")

        # Save structural summary
        structure_counts = patterns_df["structure"].value_counts()
        structure_counts.to_csv(
            self.analysis_dir / "structural_patterns_summary.csv", header=["count"]
        )

    def analyze_sequence_similarity(self):
        """Compare behavior sequences across all simulations."""
        n_sims = len(self.all_behavior_sequences)

        if n_sims < 2:
            print("⚠️  Need at least 2 simulations for similarity analysis")
            return

        print(f"🔍 Computing pairwise similarities for {n_sims} simulations...")

        similarity_results = []

        # Compute pairwise similarities (sample if too many)
        max_comparisons = 5000  # Limit to avoid excessive computation

        if n_sims * (n_sims - 1) // 2 > max_comparisons:
            # Sample pairs randomly
            print(f"   (Sampling {max_comparisons} pairs for efficiency)")
            pairs = []
            for _ in range(max_comparisons):
                i, j = random.sample(range(n_sims), 2)
                if i > j:
                    i, j = j, i
                pairs.append((i, j))
            pairs = list(set(pairs))  # Remove duplicates
        else:
            # Compute all pairs
            pairs = [(i, j) for i in range(n_sims) for j in range(i + 1, n_sims)]

        for i, j in pairs:
            seq1 = self.all_behavior_sequences[i]["sequence"]
            seq2 = self.all_behavior_sequences[j]["sequence"]
            seed1 = self.all_behavior_sequences[i]["seed"]
            seed2 = self.all_behavior_sequences[j]["seed"]

            similarity = PatternDetector.calculate_sequence_similarity(seq1, seq2)

            similarity_results.append(
                {
                    "seed1": seed1,
                    "seed2": seed2,
                    "exact_match_ratio": similarity["exact_match_ratio"],
                    "frequency_similarity": similarity["frequency_similarity"],
                    "edit_similarity": similarity["normalized_edit_similarity"],
                    "overall_similarity": similarity["overall_similarity"],
                }
            )

        # Save results
        similarity_df = pd.DataFrame(similarity_results)
        similarity_df.to_csv(self.analysis_dir / "sequence_similarity.csv", index=False)

        # Compute summary statistics
        summary = {
            "mean_overall_similarity": float(
                similarity_df["overall_similarity"].mean()
            ),
            "std_overall_similarity": float(similarity_df["overall_similarity"].std()),
            "mean_exact_match": float(similarity_df["exact_match_ratio"].mean()),
            "mean_frequency_similarity": float(
                similarity_df["frequency_similarity"].mean()
            ),
            "mean_edit_similarity": float(similarity_df["edit_similarity"].mean()),
            "num_comparisons": len(similarity_results),
            "num_simulations": n_sims,
        }

        # Save summary
        with open(self.analysis_dir / "similarity_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Create visualization
        self._plot_similarity_distribution(similarity_df)

        print(f"✓ Similarity analysis complete:")
        print(f"  - {len(similarity_results)} pairwise comparisons")
        print(f"  - Mean overall similarity: {summary['mean_overall_similarity']:.3f}")
        print(f"  - Std overall similarity: {summary['std_overall_similarity']:.3f}")

    def _plot_similarity_distribution(self, similarity_df):
        """Plot distribution of similarity scores."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics = [
            ("overall_similarity", "Overall Similarity"),
            ("exact_match_ratio", "Exact Match Ratio"),
            ("frequency_similarity", "Frequency Similarity"),
            ("edit_similarity", "Edit Distance Similarity"),
        ]

        for ax, (metric, title) in zip(axes.flat, metrics):
            ax.hist(
                similarity_df[metric],
                bins=30,
                color="steelblue",
                alpha=0.7,
                edgecolor="black",
            )
            ax.axvline(
                similarity_df[metric].mean(),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {similarity_df[metric].mean():.3f}",
            )
            ax.set_xlabel("Similarity Score", fontsize=11)
            ax.set_ylabel("Frequency", fontsize=11)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.analysis_dir / "similarity_distributions.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"✓ Similarity distribution plot saved")

    def save_metadata(self):
        """Save metadata for all simulations."""
        metadata_df = pd.DataFrame(self.simulation_metadata)
        metadata_df.to_csv(self.base_dir / "simulation_metadata.csv", index=False)

        # Create summary statistics
        summary = {
            "total_simulations": len(self.simulation_metadata),
            "simulations_with_patterns": sum(
                1 for m in self.simulation_metadata if m["has_pattern"]
            ),
            "mean_decay_rate": float(
                np.mean([m["decay_rate"] for m in self.simulation_metadata])
            ),
            "std_decay_rate": float(
                np.std([m["decay_rate"] for m in self.simulation_metadata])
            ),
        }

        with open(self.base_dir / "batch_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Metadata saved for {len(self.simulation_metadata)} simulations")


def run_batch_simulations(
    n_simulations=1000,
    n_motives=8,
    steps=2000,
    growth_rate=1,
    inter_mean=0.2,
    inter_sd=0.3,
    sat_mean=0.2,
    sat_sd=0.3,
):
    """
    Run batch simulations with optimized storage and analysis.

    Args:
        n_simulations: Number of simulations to run
        n_motives: Number of motives per simulation
        steps: Time steps per simulation
        growth_rate: Growth rate parameter
        inter_mean, inter_sd: Interaction matrix parameters
        sat_mean, sat_sd: Initial satisfaction parameters
    """
    print("=" * 70)
    print(f"BATCH SIMULATION: {n_simulations} runs")
    print("=" * 70)

    processor = BatchSimulationProcessor()

    # Run simulations
    print(f"\n🔄 Running {n_simulations} simulations...")
    import time

    start_time = time.time()

    for seed in range(n_simulations):
        if (seed + 1) % 10 == 0 or seed == 0:
            elapsed = time.time() - start_time
            per_sim = elapsed / (seed + 1) if seed > 0 else 0
            remaining = per_sim * (n_simulations - seed - 1)
            print(
                f"   Progress: {seed + 1}/{n_simulations} ({100*(seed+1)/n_simulations:.1f}%) "
                f"| Elapsed: {elapsed/60:.1f}m | Est. remaining: {remaining/60:.1f}m"
            )

        processor.run_single_simulation(
            seed=seed,
            n_motives=n_motives,
            steps=steps,
            growth_rate=growth_rate,
            inter_mean=inter_mean,
            inter_sd=inter_sd,
            sat_mean=sat_mean,
            sat_sd=sat_sd,
        )

    total_time = time.time() - start_time
    print(f"✓ All {n_simulations} simulations completed in {total_time/60:.1f} minutes")

    # Run analyses
    print("\n📊 Running analyses...")
    processor.analyze_patterns()
    processor.analyze_sequence_similarity()
    processor.save_metadata()

    print("\n" + "=" * 70)
    print("BATCH PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {processor.base_dir}")
    print("\nFolder structure:")
    print(f"  📁 {processor.base_dir}/")
    print(f"     ├── data/                    (individual simulation data)")
    print(f"     │   ├── sim_0000/")
    print(f"     │   ├── sim_0001/")
    print(f"     │   └── ...")
    print(f"     ├── plots/                   (satisfaction_over_time plots)")
    print(f"     ├── analysis/                (pattern & similarity analysis)")
    print(f"     │   ├── detected_patterns.csv")
    print(f"     │   ├── sequence_similarity.csv")
    print(f"     │   ├── similarity_summary.json")
    print(f"     │   └── similarity_distributions.png")
    print(f"     ├── simulation_metadata.csv")
    print(f"     └── batch_summary.json")
    print("\n" + "=" * 70)

    return processor


if __name__ == "__main__":
    # Run batch processing
    processor = run_batch_simulations(
        n_simulations=1000,
        n_motives=8,
        steps=1000,
        growth_rate=1,
        inter_mean=0.1,
        inter_sd=0.6,
        sat_mean=0.1,
        sat_sd=0.2,
    )

    print("\n✅ Batch processing completed successfully!")
    print(f"📂 Check results in: {processor.base_dir}")

"""
Comprehensive visualization and analysis module for the motive game engine.
Generates graphs, metrics, and saves all outputs to timestamped folders.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import json
from collections import Counter
from scipy import stats


class MotiveAnalyzer:
    """Analyzes game engine output and generates comprehensive visualizations."""

    def __init__(self, history, inter_m, sat_m_initial, params):
        """
        Initialize analyzer with simulation results.

        Args:
            history: Dictionary from game_engine with keys: step, active_behavior, satisfaction, unsatisfied_octants
            inter_m: Interaction matrix (DataFrame)
            sat_m_initial: Initial satisfaction matrix (DataFrame)
            params: Dictionary of simulation parameters (steps, decay_rate, growth_rate, etc.)
        """
        self.history = history
        self.inter_m = inter_m
        self.sat_m_initial = sat_m_initial
        self.params = params
        self.motive_names = list(inter_m.columns)
        self.n_motives = len(self.motive_names)

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"simulation_results_{timestamp}")
        self.output_dir.mkdir(exist_ok=True)

        # Extract satisfaction history as array
        self.satisfaction_array = self._extract_satisfaction_array()
        self.behavior_sequence = history["active_behavior"]

    def _extract_satisfaction_array(self):
        """Convert satisfaction history to numpy array for easier analysis."""
        n_steps = len(self.history["step"])
        sat_array = np.zeros((n_steps, self.n_motives))

        for i, sat_df in enumerate(self.history["satisfaction"]):
            sat_array[i, :] = sat_df.loc["satisfaction"].values

        return sat_array

    def run_full_analysis(self):
        """Run all analyses and save all outputs."""
        print(f"Starting analysis... Output directory: {self.output_dir}")

        # 1. Plot satisfaction over time
        self.plot_satisfaction_over_time()

        # 2. Plot active behaviors over time
        self.plot_active_behaviors()

        # 3. Calculate and save all metrics
        metrics = self.calculate_all_metrics()

        # 4. Save parameters and metrics as CSV
        self.save_analysis_data(metrics)

        # 5. Create comprehensive plots
        self.plot_behavior_analysis()
        self.plot_network_analysis()
        self.plot_markov_chain()

        print(f"✓ Analysis complete! Results saved to: {self.output_dir}")
        return metrics

    def plot_satisfaction_over_time(self):
        """Plot 1: 8 motives and their satisfaction over time with y=0 red line."""
        fig, ax = plt.subplots(figsize=(14, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, self.n_motives))

        for i, motive in enumerate(self.motive_names):
            ax.plot(
                self.history["step"],
                self.satisfaction_array[:, i],
                label=motive,
                color=colors[i],
                linewidth=2,
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

        # Grey lines at boundaries
        ax.axhline(y=1, color="grey", linestyle="--", linewidth=1, alpha=0.5)
        ax.axhline(y=-1, color="grey", linestyle="--", linewidth=1, alpha=0.5)

        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel("Satisfaction Level", fontsize=12)
        ax.set_title("Motive Satisfaction Over Time", fontsize=14, fontweight="bold")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(
            self.output_dir / f"satisfaction_over_time_{timestamp}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print("✓ Saved: satisfaction_over_time")

    def plot_active_behaviors(self):
        """Plot 2: Active behaviors over time with larger markers for consecutive activations."""
        fig, ax = plt.subplots(figsize=(14, 6))

        # Create numerical encoding for behaviors
        behavior_to_num = {behavior: i for i, behavior in enumerate(self.motive_names)}
        behavior_to_num[None] = -1  # None behaviors

        # Convert behaviors to numbers
        behavior_nums = [behavior_to_num[b] for b in self.behavior_sequence]

        # Identify consecutive activations (same behavior twice in a row)
        consecutive_mask = np.zeros(len(behavior_nums), dtype=bool)
        for i in range(1, len(behavior_nums)):
            if behavior_nums[i] == behavior_nums[i - 1] and behavior_nums[i] != -1:
                consecutive_mask[i] = True
                consecutive_mask[i - 1] = True

        # Plot with different sizes
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_motives))

        for i, behavior in enumerate(self.behavior_sequence):
            if behavior is not None:
                behavior_idx = self.motive_names.index(behavior)
                size = 150 if consecutive_mask[i] else 50
                alpha = 0.9 if consecutive_mask[i] else 0.5
                ax.scatter(
                    i,
                    behavior_idx,
                    s=size,
                    color=colors[behavior_idx],
                    alpha=alpha,
                    edgecolors="black",
                    linewidth=0.5,
                )

        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel("Active Behavior", fontsize=12)
        ax.set_title(
            "Active Behaviors Over Time\n(Larger markers = consecutive activations)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_yticks(range(self.n_motives))
        ax.set_yticklabels(self.motive_names)
        ax.grid(True, alpha=0.3, axis="x")
        ax.set_ylim(-0.5, self.n_motives - 0.5)

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(
            self.output_dir / f"active_behaviors_{timestamp}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print("✓ Saved: active_behaviors")

    def calculate_all_metrics(self):
        """Calculate all requested input/output metrics."""
        metrics = {}

        # ===== INPUT METRICS =====
        metrics["input"] = {}

        # 1. Mean congruence (mean of interaction matrix, excluding diagonal)
        inter_values = self.inter_m.values
        mask = ~np.eye(self.n_motives, dtype=bool)
        metrics["input"]["mean_congruence"] = float(np.mean(inter_values[mask]))

        # 2. Mean conflict (mean of negative values in interaction matrix)
        negative_values = inter_values[mask][inter_values[mask] < 0]
        metrics["input"]["mean_conflict"] = (
            float(np.mean(negative_values)) if len(negative_values) > 0 else 0.0
        )

        # 3. Mean initial satisfaction
        metrics["input"]["mean_initial_satisfaction"] = float(
            self.sat_m_initial.loc["satisfaction"].mean()
        )

        # 4. Network structure metrics
        metrics["input"]["network"] = self._analyze_network_structure()

        # ===== OUTPUT METRICS =====
        metrics["output"] = {}

        # 1. Mean satisfaction level (averaged over time)
        metrics["output"]["mean_satisfaction_over_time"] = float(
            np.mean(self.satisfaction_array)
        )
        metrics["output"]["final_mean_satisfaction"] = float(
            np.mean(self.satisfaction_array[-1, :])
        )

        # 2. Behavior rhythm (time between behavior switches)
        metrics["output"]["behavior_rhythm"] = self._calculate_behavior_rhythm()

        # 3. Behavior frequencies
        metrics["output"][
            "behavior_frequencies"
        ] = self._calculate_behavior_frequencies()

        # 4. Dominant behavior (mean activation level per behavior)
        metrics["output"]["dominant_behaviors"] = self._calculate_dominant_behaviors()

        # 5. Time until same behavior reappears
        metrics["output"][
            "behavior_reappearance_times"
        ] = self._calculate_reappearance_times()

        # 6. Satisfaction duration statistics
        metrics["output"][
            "satisfaction_durations"
        ] = self._calculate_satisfaction_durations()

        # 7. Multiple simultaneous (un)satisfactions
        metrics["output"][
            "simultaneous_satisfactions"
        ] = self._calculate_simultaneous_satisfactions()

        # 8. Markov chain analysis
        metrics["output"]["markov_chain"] = self._calculate_markov_chain()

        return metrics

    def _analyze_network_structure(self):
        """Analyze the interaction matrix network structure."""
        network_metrics = {}

        # Find strongest dyads (pairs with highest absolute interaction)
        inter_values = self.inter_m.values
        n = self.n_motives

        # Get upper triangle (excluding diagonal) to avoid duplicates
        dyad_strengths = []
        for i in range(n):
            for j in range(i + 1, n):
                strength = abs(inter_values[i, j])
                dyad_strengths.append(
                    {
                        "dyad": f"{self.motive_names[i]}-{self.motive_names[j]}",
                        "interaction": float(inter_values[i, j]),
                        "strength": float(strength),
                    }
                )

        # Sort by strength and get top 5
        dyad_strengths.sort(key=lambda x: x["strength"], reverse=True)
        network_metrics["top_dyads"] = dyad_strengths[:5]

        # Network density (proportion of strong connections)
        threshold = 0.3  # Define "strong" connection
        strong_connections = np.sum(
            np.abs(inter_values[~np.eye(n, dtype=bool)]) > threshold
        )
        total_connections = n * (n - 1)
        network_metrics["density"] = float(strong_connections / total_connections)

        # Positive vs negative interaction balance
        mask = ~np.eye(n, dtype=bool)
        network_metrics["positive_ratio"] = float(np.mean(inter_values[mask] > 0))

        return network_metrics

    def _calculate_behavior_rhythm(self):
        """Calculate time between behavior switches."""
        switches = []
        current_behavior = None
        time_in_behavior = 0

        for behavior in self.behavior_sequence:
            if behavior != current_behavior:
                if current_behavior is not None:
                    switches.append(time_in_behavior)
                current_behavior = behavior
                time_in_behavior = 1
            else:
                time_in_behavior += 1

        # Add final duration
        if time_in_behavior > 0:
            switches.append(time_in_behavior)

        return {
            "mean_duration": float(np.mean(switches)) if switches else 0.0,
            "std_duration": float(np.std(switches)) if switches else 0.0,
            "median_duration": float(np.median(switches)) if switches else 0.0,
            "switch_count": len(switches),
        }

    def _calculate_behavior_frequencies(self):
        """Count how often each behavior was active."""
        active_behaviors = [b for b in self.behavior_sequence if b is not None]
        counts = Counter(active_behaviors)

        frequencies = {}
        for motive in self.motive_names:
            frequencies[motive] = counts.get(motive, 0)

        return frequencies

    def _calculate_dominant_behaviors(self):
        """Calculate mean 'activation level' for each behavior (spider diagram data)."""
        # For each behavior, calculate what proportion of time it was active
        total_steps = len(self.behavior_sequence)
        activation_levels = {}

        for motive in self.motive_names:
            count = sum(1 for b in self.behavior_sequence if b == motive)
            activation_levels[motive] = float(count / total_steps)

        return activation_levels

    def _calculate_reappearance_times(self):
        """Calculate how long until same behavior reappears."""
        reappearance_times = {motive: [] for motive in self.motive_names}
        last_seen = {motive: None for motive in self.motive_names}

        for step, behavior in enumerate(self.behavior_sequence):
            if behavior is not None:
                if last_seen[behavior] is not None:
                    time_since = step - last_seen[behavior]
                    reappearance_times[behavior].append(time_since)
                last_seen[behavior] = step

        # Calculate statistics
        stats_dict = {}
        for motive in self.motive_names:
            times = reappearance_times[motive]
            if times:
                stats_dict[motive] = {
                    "mean": float(np.mean(times)),
                    "std": float(np.std(times)),
                    "min": int(np.min(times)),
                    "max": int(np.max(times)),
                }
            else:
                stats_dict[motive] = None

        return stats_dict

    def _calculate_satisfaction_durations(self):
        """Calculate how long person stayed satisfied/unsatisfied."""
        # A person is satisfied if ANY motive is satisfied (>0)
        # Track continuous periods of satisfaction

        satisfied_periods = []
        unsatisfied_periods = []

        current_state = None  # 'satisfied' or 'unsatisfied'
        duration = 0

        for step in range(len(self.satisfaction_array)):
            any_satisfied = np.any(self.satisfaction_array[step, :] > 0)
            state = "satisfied" if any_satisfied else "unsatisfied"

            if state != current_state:
                if current_state is not None:
                    if current_state == "satisfied":
                        satisfied_periods.append(duration)
                    else:
                        unsatisfied_periods.append(duration)
                current_state = state
                duration = 1
            else:
                duration += 1

        # Add final period
        if current_state == "satisfied":
            satisfied_periods.append(duration)
        else:
            unsatisfied_periods.append(duration)

        return {
            "satisfied": {
                "mean": float(np.mean(satisfied_periods)) if satisfied_periods else 0.0,
                "total_time": (
                    int(np.sum(satisfied_periods)) if satisfied_periods else 0
                ),
                "num_periods": len(satisfied_periods),
            },
            "unsatisfied": {
                "mean": (
                    float(np.mean(unsatisfied_periods)) if unsatisfied_periods else 0.0
                ),
                "total_time": (
                    int(np.sum(unsatisfied_periods)) if unsatisfied_periods else 0
                ),
                "num_periods": len(unsatisfied_periods),
            },
        }

    def _calculate_simultaneous_satisfactions(self):
        """Calculate how often multiple motives were (un)satisfied simultaneously."""
        simultaneous_stats = {"satisfied": [], "unsatisfied": []}

        for step in range(len(self.satisfaction_array)):
            n_satisfied = np.sum(self.satisfaction_array[step, :] > 0)
            n_unsatisfied = np.sum(self.satisfaction_array[step, :] < 0)

            simultaneous_stats["satisfied"].append(int(n_satisfied))
            simultaneous_stats["unsatisfied"].append(int(n_unsatisfied))

        return {
            "satisfied": {
                "mean": float(np.mean(simultaneous_stats["satisfied"])),
                "max": int(np.max(simultaneous_stats["satisfied"])),
                "distribution": dict(Counter(simultaneous_stats["satisfied"])),
            },
            "unsatisfied": {
                "mean": float(np.mean(simultaneous_stats["unsatisfied"])),
                "max": int(np.max(simultaneous_stats["unsatisfied"])),
                "distribution": dict(Counter(simultaneous_stats["unsatisfied"])),
            },
        }

    def _calculate_markov_chain(self):
        """Calculate Markov chain transition probabilities."""
        # Remove None values and create transition matrix
        active_behaviors = [b for b in self.behavior_sequence if b is not None]

        if len(active_behaviors) < 2:
            return None

        # Count transitions
        transition_counts = np.zeros((self.n_motives, self.n_motives))

        for i in range(len(active_behaviors) - 1):
            from_idx = self.motive_names.index(active_behaviors[i])
            to_idx = self.motive_names.index(active_behaviors[i + 1])
            transition_counts[from_idx, to_idx] += 1

        # Convert to probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_probs = transition_counts / row_sums

        # Convert to serializable format
        markov_dict = {}
        for i, from_motive in enumerate(self.motive_names):
            markov_dict[from_motive] = {}
            for j, to_motive in enumerate(self.motive_names):
                if transition_probs[i, j] > 0:
                    markov_dict[from_motive][to_motive] = float(transition_probs[i, j])

        return {
            "transition_probabilities": markov_dict,
            "transition_matrix": transition_probs.tolist(),
        }

    def save_analysis_data(self, metrics):
        """Save all parameters, matrices, and metrics as CSV files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Save parameters
        params_df = pd.DataFrame([self.params])
        params_df.to_csv(self.output_dir / f"parameters_{timestamp}.csv", index=False)

        # 2. Save interaction matrix
        self.inter_m.to_csv(self.output_dir / f"interaction_matrix_{timestamp}.csv")

        # 3. Save initial satisfaction
        self.sat_m_initial.to_csv(
            self.output_dir / f"initial_satisfaction_{timestamp}.csv"
        )

        # 4. Save metrics as JSON (more flexible for nested structure)
        with open(self.output_dir / f"metrics_{timestamp}.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # 5. Save flattened metrics as CSV for easy viewing
        flat_metrics = self._flatten_metrics(metrics)
        flat_df = pd.DataFrame([flat_metrics])
        flat_df.to_csv(
            self.output_dir / f"metrics_summary_{timestamp}.csv", index=False
        )

        # 6. Save behavior sequence
        behavior_df = pd.DataFrame(
            {"step": self.history["step"], "active_behavior": self.behavior_sequence}
        )
        behavior_df.to_csv(
            self.output_dir / f"behavior_sequence_{timestamp}.csv", index=False
        )

        # 7. Save satisfaction time series
        sat_df = pd.DataFrame(self.satisfaction_array, columns=self.motive_names)
        sat_df.insert(0, "step", self.history["step"])
        sat_df.to_csv(
            self.output_dir / f"satisfaction_timeseries_{timestamp}.csv", index=False
        )

        print("✓ Saved: all data files (CSV, JSON)")

    def _flatten_metrics(self, metrics, parent_key="", sep="_"):
        """Flatten nested dictionary for CSV export."""
        items = []
        for k, v in metrics.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_metrics(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def plot_behavior_analysis(self):
        """Create comprehensive behavior analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Behavior frequencies (bar chart)
        ax = axes[0, 0]
        freqs = self._calculate_behavior_frequencies()
        ax.bar(
            self.motive_names,
            [freqs[m] for m in self.motive_names],
            color=plt.cm.tab10(np.linspace(0, 1, self.n_motives)),
        )
        ax.set_xlabel("Behavior")
        ax.set_ylabel("Frequency")
        ax.set_title("Behavior Activation Frequencies", fontweight="bold")
        ax.tick_params(axis="x", rotation=45)

        # 2. Dominant behaviors (radar/spider chart)
        ax = axes[0, 1]
        dominant = self._calculate_dominant_behaviors()
        values = [dominant[m] for m in self.motive_names]

        angles = np.linspace(0, 2 * np.pi, self.n_motives, endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]

        ax_polar = plt.subplot(2, 2, 2, projection="polar")
        ax_polar.plot(angles, values, "o-", linewidth=2, color="steelblue")
        ax_polar.fill(angles, values, alpha=0.25, color="steelblue")
        ax_polar.set_xticks(angles[:-1])
        ax_polar.set_xticklabels(self.motive_names)
        ax_polar.set_ylim(0, max(values) * 1.1)
        ax_polar.set_title(
            "Dominant Behaviors (Activation Proportions)", fontweight="bold", pad=20
        )
        axes[0, 1].remove()  # Remove the cartesian axes

        # 3. Satisfaction duration histogram
        ax = axes[1, 0]
        durations = self._calculate_satisfaction_durations()
        categories = ["Satisfied\nPeriods", "Unsatisfied\nPeriods"]
        means = [durations["satisfied"]["mean"], durations["unsatisfied"]["mean"]]
        colors_dur = ["green", "red"]

        bars = ax.bar(categories, means, color=colors_dur, alpha=0.7)
        ax.set_ylabel("Mean Duration (steps)")
        ax.set_title(
            "Mean Duration of Satisfied vs Unsatisfied States", fontweight="bold"
        )

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
            )

        # 4. Simultaneous (un)satisfactions over time
        ax = axes[1, 1]
        simul_stats = self._calculate_simultaneous_satisfactions()

        # Plot counts over time
        satisfied_counts = []
        unsatisfied_counts = []
        for step in range(len(self.satisfaction_array)):
            satisfied_counts.append(np.sum(self.satisfaction_array[step, :] > 0))
            unsatisfied_counts.append(np.sum(self.satisfaction_array[step, :] < 0))

        ax.plot(
            self.history["step"],
            satisfied_counts,
            label="Satisfied",
            color="green",
            alpha=0.7,
            linewidth=2,
        )
        ax.plot(
            self.history["step"],
            unsatisfied_counts,
            label="Unsatisfied",
            color="red",
            alpha=0.7,
            linewidth=2,
        )
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Number of Motives")
        ax.set_title("Simultaneous (Un)Satisfactions Over Time", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"behavior_analysis_{timestamp}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print("✓ Saved: behavior_analysis")

    def plot_network_analysis(self):
        """Visualize the interaction matrix network structure."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Heatmap of interaction matrix
        ax = axes[0]
        sns.heatmap(
            self.inter_m,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax,
            cbar_kws={"label": "Interaction Strength"},
        )
        ax.set_title("Interaction Matrix Heatmap", fontweight="bold", fontsize=14)

        # 2. Network graph of strongest connections
        ax = axes[1]

        # Get top dyads
        network_metrics = self._analyze_network_structure()
        top_dyads = network_metrics["top_dyads"][:10]  # Top 10 connections

        # Create positions in circle
        angles = np.linspace(0, 2 * np.pi, self.n_motives, endpoint=False)
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)

        # Plot nodes
        ax.scatter(
            x_pos,
            y_pos,
            s=500,
            c="lightblue",
            edgecolors="black",
            linewidth=2,
            zorder=3,
        )

        # Add labels
        for i, motive in enumerate(self.motive_names):
            ax.text(
                x_pos[i] * 1.15,
                y_pos[i] * 1.15,
                motive,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )

        # Draw edges for top connections
        for dyad_info in top_dyads:
            dyad = dyad_info["dyad"]
            m1, m2 = dyad.split("-")
            i1 = self.motive_names.index(m1)
            i2 = self.motive_names.index(m2)

            interaction = dyad_info["interaction"]
            color = "green" if interaction > 0 else "red"
            linewidth = abs(interaction) * 5  # Scale line width

            ax.plot(
                [x_pos[i1], x_pos[i2]],
                [y_pos[i1], y_pos[i2]],
                color=color,
                linewidth=linewidth,
                alpha=0.6,
                zorder=1,
            )

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            "Network of Strongest Interactions\n(Green=Positive, Red=Negative)",
            fontweight="bold",
            fontsize=14,
        )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"network_analysis_{timestamp}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print("✓ Saved: network_analysis")

    def plot_markov_chain(self):
        """Visualize Markov chain transition probabilities."""
        markov_data = self._calculate_markov_chain()

        if markov_data is None:
            print("⚠ Skipping Markov chain plot (insufficient data)")
            return

        transition_matrix = np.array(markov_data["transition_matrix"])

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Transition matrix heatmap
        ax = axes[0]
        sns.heatmap(
            transition_matrix,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            xticklabels=self.motive_names,
            yticklabels=self.motive_names,
            square=True,
            ax=ax,
            cbar_kws={"label": "Transition Probability"},
        )
        ax.set_xlabel("To Behavior")
        ax.set_ylabel("From Behavior")
        ax.set_title(
            "Markov Chain Transition Probabilities", fontweight="bold", fontsize=14
        )

        # 2. Network diagram of transitions
        ax = axes[1]

        # Positions in circle
        angles = np.linspace(0, 2 * np.pi, self.n_motives, endpoint=False)
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)

        # Plot nodes
        ax.scatter(
            x_pos,
            y_pos,
            s=500,
            c="lightcoral",
            edgecolors="black",
            linewidth=2,
            zorder=3,
        )

        # Add labels
        for i, motive in enumerate(self.motive_names):
            ax.text(
                x_pos[i] * 1.15,
                y_pos[i] * 1.15,
                motive,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )

        # Draw arrows for significant transitions (prob > 0.1)
        for i in range(self.n_motives):
            for j in range(self.n_motives):
                if i != j and transition_matrix[i, j] > 0.1:
                    # Draw arrow from i to j
                    dx = x_pos[j] - x_pos[i]
                    dy = y_pos[j] - y_pos[i]

                    # Shorten arrow to not overlap with nodes
                    length = np.sqrt(dx**2 + dy**2)
                    dx_norm = dx / length * 0.15  # Offset from node
                    dy_norm = dy / length * 0.15

                    ax.arrow(
                        x_pos[i] + dx_norm,
                        y_pos[i] + dy_norm,
                        dx - 2 * dx_norm,
                        dy - 2 * dy_norm,
                        head_width=0.05,
                        head_length=0.05,
                        fc="gray",
                        ec="gray",
                        alpha=transition_matrix[i, j],
                        linewidth=transition_matrix[i, j] * 3,
                        zorder=2,
                    )

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            "Markov Chain State Diagram\n(Arrow darkness = transition probability)",
            fontweight="bold",
            fontsize=14,
        )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"markov_chain_{timestamp}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print("✓ Saved: markov_chain")


def visualize_simulation(history, inter_m, sat_m_initial, params):
    """
    Convenience function to run full analysis.

    Args:
        history: Output from game_engine()
        inter_m: Interaction matrix
        sat_m_initial: Initial satisfaction matrix
        params: Dictionary with keys: steps, decay_rate, growth_rate, etc.

    Returns:
        MotiveAnalyzer instance with all results
    """
    analyzer = MotiveAnalyzer(history, inter_m, sat_m_initial, params)
    analyzer.run_full_analysis()
    return analyzer

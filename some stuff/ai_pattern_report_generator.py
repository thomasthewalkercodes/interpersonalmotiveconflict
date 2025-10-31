"""
Comprehensive Pattern Analysis Report Generator
Creates a visually appealing analysis document with:
1. Simulations with similar patterns
2. Pattern length distribution
3. Most common patterns
4. Structural pattern analysis
5. And more insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import matplotlib.patches as mpatches
from datetime import datetime

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class PatternAnalysisReport:
    """Generate comprehensive visual analysis report from batch simulation results."""

    def __init__(self, batch_results_dir):
        """
        Initialize report generator.

        Args:
            batch_results_dir: Path to batch_simulations_* directory
        """
        self.base_dir = Path(batch_results_dir)
        self.analysis_dir = self.base_dir / "analysis"
        self.report_dir = self.base_dir / "pattern_report"
        self.report_dir.mkdir(exist_ok=True)

        # Load data
        self.load_data()

    def load_data(self):
        """Load all necessary data files."""
        print("📂 Loading data...")

        # Load patterns
        patterns_file = self.analysis_dir / "detected_patterns.csv"
        if patterns_file.exists():
            self.patterns_df = pd.read_csv(patterns_file)
            print(f"   ✓ Loaded {len(self.patterns_df)} patterns")
        else:
            self.patterns_df = pd.DataFrame()
            print("   ⚠️  No patterns file found")

        # Load metadata
        metadata_file = self.base_dir / "simulation_metadata.csv"
        if metadata_file.exists():
            self.metadata_df = pd.read_csv(metadata_file)
            print(f"   ✓ Loaded metadata for {len(self.metadata_df)} simulations")
        else:
            self.metadata_df = pd.DataFrame()
            print("   ⚠️  No metadata file found")

        # Load similarity if exists
        similarity_file = self.analysis_dir / "sequence_similarity.csv"
        if similarity_file.exists():
            self.similarity_df = pd.read_csv(similarity_file)
            print(f"   ✓ Loaded {len(self.similarity_df)} similarity comparisons")
        else:
            self.similarity_df = pd.DataFrame()
            print("   ⚠️  No similarity file found")

    def generate_full_report(self):
        """Generate complete analysis report with all visualizations."""
        print("\n" + "=" * 70)
        print("GENERATING COMPREHENSIVE PATTERN ANALYSIS REPORT")
        print("=" * 70)

        # Create all figures
        self.create_overview_figure()
        self.create_pattern_length_distribution()
        self.create_most_common_patterns()
        self.create_structural_analysis()
        self.create_similarity_matrix()
        self.create_pattern_occurrence_analysis()
        self.save_summary_statistics()

        print("\n" + "=" * 70)
        print(f"✓ REPORT COMPLETE: {self.report_dir}")
        print("=" * 70)
        print("\nGenerated files:")
        for f in sorted(self.report_dir.glob("*.png")):
            print(f"  📊 {f.name}")
        for f in sorted(self.report_dir.glob("*.csv")):
            print(f"  📄 {f.name}")
        print("\n" + "=" * 70)

    def create_overview_figure(self):
        """Create 4-panel overview figure."""
        if self.patterns_df.empty:
            print("⚠️  Skipping overview (no patterns)")
            return

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Panel 1: Pattern detection rate
        ax1 = fig.add_subplot(gs[0, 0])
        has_pattern_count = self.metadata_df["has_pattern"].sum()
        total_sims = len(self.metadata_df)

        colors_pie = ["#2ecc71", "#e74c3c"]
        ax1.pie(
            [has_pattern_count, total_sims - has_pattern_count],
            labels=["Has Pattern", "No Pattern"],
            autopct="%1.1f%%",
            colors=colors_pie,
            startangle=90,
            textprops={"fontsize": 12, "weight": "bold"},
        )
        ax1.set_title(
            f"Pattern Detection Rate\n({has_pattern_count}/{total_sims} simulations)",
            fontsize=14,
            weight="bold",
            pad=20,
        )

        # Panel 2: Pattern length histogram
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(
            self.patterns_df["pattern_length"],
            bins=20,
            color="steelblue",
            edgecolor="black",
            alpha=0.7,
        )
        ax2.axvline(
            self.patterns_df["pattern_length"].mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {self.patterns_df['pattern_length'].mean():.1f}",
        )
        ax2.axvline(
            self.patterns_df["pattern_length"].median(),
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"Median: {self.patterns_df['pattern_length'].median():.1f}",
        )
        ax2.set_xlabel("Pattern Length (# of behaviors)", fontsize=11)
        ax2.set_ylabel("Frequency", fontsize=11)
        ax2.set_title("Distribution of Pattern Lengths", fontsize=14, weight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Panel 3: Pattern occurrences
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(
            self.patterns_df["num_occurrences"],
            bins=20,
            color="coral",
            edgecolor="black",
            alpha=0.7,
        )
        ax3.axvline(
            self.patterns_df["num_occurrences"].mean(),
            color="darkred",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {self.patterns_df['num_occurrences'].mean():.1f}",
        )
        ax3.set_xlabel("Number of Pattern Occurrences", fontsize=11)
        ax3.set_ylabel("Frequency", fontsize=11)
        ax3.set_title("Pattern Repetition Frequency", fontsize=14, weight="bold")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Panel 4: Structural patterns (top 10)
        ax4 = fig.add_subplot(gs[1, 1])
        if "structure" in self.patterns_df.columns:
            structure_counts = self.patterns_df["structure"].value_counts().head(10)
            ax4.barh(
                range(len(structure_counts)),
                structure_counts.values,
                color="mediumpurple",
                edgecolor="black",
                alpha=0.7,
            )
            ax4.set_yticks(range(len(structure_counts)))
            ax4.set_yticklabels(structure_counts.index, fontsize=9)
            ax4.set_xlabel("Frequency", fontsize=11)
            ax4.set_title("Top 10 Structural Patterns", fontsize=14, weight="bold")
            ax4.grid(True, alpha=0.3, axis="x")
            ax4.invert_yaxis()

        # Panel 5: Pattern length vs occurrences scatter
        ax5 = fig.add_subplot(gs[2, :])
        scatter = ax5.scatter(
            self.patterns_df["pattern_length"],
            self.patterns_df["num_occurrences"],
            c=self.patterns_df["seed"],
            cmap="viridis",
            alpha=0.6,
            s=50,
            edgecolors="black",
            linewidth=0.5,
        )
        ax5.set_xlabel("Pattern Length", fontsize=12)
        ax5.set_ylabel("Number of Occurrences", fontsize=12)
        ax5.set_title(
            "Pattern Length vs Repetition Frequency (colored by simulation seed)",
            fontsize=14,
            weight="bold",
        )
        ax5.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax5, label="Simulation Seed")

        plt.suptitle("Pattern Analysis Overview", fontsize=18, weight="bold", y=0.995)

        plt.savefig(self.report_dir / "01_overview.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("✓ Created: 01_overview.png")

    def create_pattern_length_distribution(self):
        """Detailed pattern length distribution analysis."""
        if self.patterns_df.empty:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # 1. Detailed histogram
        ax = axes[0, 0]
        ax.hist(
            self.patterns_df["pattern_length"],
            bins=30,
            color="skyblue",
            edgecolor="black",
            alpha=0.7,
            density=True,
        )

        # Add KDE
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(self.patterns_df["pattern_length"])
        x_range = np.linspace(
            self.patterns_df["pattern_length"].min(),
            self.patterns_df["pattern_length"].max(),
            100,
        )
        ax.plot(x_range, kde(x_range), "r-", linewidth=2, label="KDE")

        ax.set_xlabel("Pattern Length", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(
            "Pattern Length Distribution (with KDE)", fontsize=12, weight="bold"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Box plot
        ax = axes[0, 1]
        bp = ax.boxplot(
            [self.patterns_df["pattern_length"]],
            vert=True,
            patch_artist=True,
            widths=0.6,
        )
        bp["boxes"][0].set_facecolor("lightcoral")
        bp["boxes"][0].set_alpha(0.7)

        ax.set_ylabel("Pattern Length", fontsize=11)
        ax.set_title("Pattern Length Box Plot", fontsize=12, weight="bold")
        ax.set_xticklabels(["All Patterns"])
        ax.grid(True, alpha=0.3, axis="y")

        # Add statistics text
        stats_text = f"Mean: {self.patterns_df['pattern_length'].mean():.2f}\n"
        stats_text += f"Median: {self.patterns_df['pattern_length'].median():.2f}\n"
        stats_text += f"Std: {self.patterns_df['pattern_length'].std():.2f}\n"
        stats_text += f"Min: {self.patterns_df['pattern_length'].min():.0f}\n"
        stats_text += f"Max: {self.patterns_df['pattern_length'].max():.0f}"
        ax.text(
            1.15,
            0.5,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # 3. Cumulative distribution
        ax = axes[1, 0]
        sorted_lengths = np.sort(self.patterns_df["pattern_length"])
        cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
        ax.plot(sorted_lengths, cumulative, linewidth=2, color="darkgreen")
        ax.set_xlabel("Pattern Length", fontsize=11)
        ax.set_ylabel("Cumulative Probability", fontsize=11)
        ax.set_title(
            "Cumulative Distribution of Pattern Lengths", fontsize=12, weight="bold"
        )
        ax.grid(True, alpha=0.3)

        # Add percentile lines
        for percentile in [25, 50, 75]:
            value = np.percentile(self.patterns_df["pattern_length"], percentile)
            ax.axvline(
                value,
                color="red",
                linestyle="--",
                alpha=0.5,
                label=f"{percentile}th: {value:.1f}",
            )
        ax.legend()

        # 4. Frequency table
        ax = axes[1, 1]
        ax.axis("off")

        # Create frequency table
        length_counts = (
            self.patterns_df["pattern_length"].value_counts().sort_index().head(15)
        )
        table_data = []
        for length, count in length_counts.items():
            percentage = count / len(self.patterns_df) * 100
            table_data.append([int(length), int(count), f"{percentage:.1f}%"])

        table = ax.table(
            cellText=table_data,
            colLabels=["Length", "Count", "Percentage"],
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header
        for i in range(3):
            table[(0, i)].set_facecolor("#4CAF50")
            table[(0, i)].set_text_props(weight="bold", color="white")

        ax.set_title(
            "Pattern Length Frequency Table (Top 15)",
            fontsize=12,
            weight="bold",
            pad=20,
        )

        plt.suptitle("Detailed Pattern Length Analysis", fontsize=16, weight="bold")
        plt.tight_layout()

        plt.savefig(
            self.report_dir / "02_pattern_lengths.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("✓ Created: 02_pattern_lengths.png")

    def create_most_common_patterns(self):
        """Analyze and visualize most common patterns."""
        if self.patterns_df.empty or "structure" not in self.patterns_df.columns:
            return

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # 1. Top 20 structural patterns
        ax = axes[0, 0]
        structure_counts = self.patterns_df["structure"].value_counts().head(20)

        y_pos = np.arange(len(structure_counts))
        bars = ax.barh(
            y_pos, structure_counts.values, color="teal", alpha=0.7, edgecolor="black"
        )

        # Color the top 5 differently
        for i in range(min(5, len(bars))):
            bars[i].set_color("darkred")
            bars[i].set_alpha(0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(structure_counts.index, fontsize=10)
        ax.set_xlabel("Frequency", fontsize=12)
        ax.set_title(
            "Top 20 Most Common Structural Patterns", fontsize=13, weight="bold"
        )
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")

        # 2. Pattern diversity pie chart
        ax = axes[0, 1]

        # Count unique structures
        unique_structures = self.patterns_df["structure"].nunique()
        total_patterns = len(self.patterns_df)

        # Group into categories
        top_10_count = self.patterns_df["structure"].value_counts().head(10).sum()
        other_count = total_patterns - top_10_count

        sizes = [top_10_count, other_count]
        labels = [
            f"Top 10 patterns\n({top_10_count})",
            f"Other patterns\n({other_count})",
        ]
        colors_diversity = ["#3498db", "#95a5a6"]

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            colors=colors_diversity,
            startangle=90,
            textprops={"fontsize": 11},
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_weight("bold")

        ax.set_title(
            f"Pattern Diversity\n({unique_structures} unique structural patterns)",
            fontsize=13,
            weight="bold",
        )

        # 3. Simulations sharing patterns
        ax = axes[1, 0]

        # Find which simulations have the same structural pattern
        structure_to_sims = {}
        for _, row in self.patterns_df.iterrows():
            struct = row["structure"]
            seed = row["seed"]
            if struct not in structure_to_sims:
                structure_to_sims[struct] = []
            structure_to_sims[struct].append(seed)

        # Count how many simulations share each pattern
        shared_counts = [len(sims) for sims in structure_to_sims.values()]
        shared_hist = Counter(shared_counts)

        x_vals = sorted(shared_hist.keys())
        y_vals = [shared_hist[x] for x in x_vals]

        ax.bar(x_vals, y_vals, color="mediumseagreen", alpha=0.7, edgecolor="black")
        ax.set_xlabel("Number of Simulations Sharing Pattern", fontsize=11)
        ax.set_ylabel("Number of Patterns", fontsize=11)
        ax.set_title("Pattern Sharing Across Simulations", fontsize=13, weight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # 4. Table of patterns with most simulations
        ax = axes[1, 1]
        ax.axis("off")

        # Get patterns shared by most simulations
        shared_patterns = [(struct, sims) for struct, sims in structure_to_sims.items()]
        shared_patterns.sort(key=lambda x: len(x[1]), reverse=True)

        table_data = []
        for i, (struct, sims) in enumerate(shared_patterns[:15]):
            sim_ids = ", ".join([str(s) for s in sorted(sims)[:5]])
            if len(sims) > 5:
                sim_ids += f", ... ({len(sims)} total)"
            table_data.append(
                [
                    struct[:25] + "..." if len(struct) > 25 else struct,
                    len(sims),
                    sim_ids[:40] + "..." if len(sim_ids) > 40 else sim_ids,
                ]
            )

        table = ax.table(
            cellText=table_data,
            colLabels=["Pattern Structure", "# Sims", "Simulation IDs"],
            cellLoc="left",
            colWidths=[0.35, 0.15, 0.5],
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)

        # Style header
        for i in range(3):
            table[(0, i)].set_facecolor("#FF5733")
            table[(0, i)].set_text_props(weight="bold", color="white")

        ax.set_title(
            "Patterns Shared by Multiple Simulations (Top 15)",
            fontsize=13,
            weight="bold",
            pad=20,
        )

        plt.suptitle("Most Common Patterns Analysis", fontsize=16, weight="bold")
        plt.tight_layout()

        plt.savefig(
            self.report_dir / "03_common_patterns.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("✓ Created: 03_common_patterns.png")

        # Save detailed CSV
        sharing_data = []
        for struct, sims in shared_patterns:
            sharing_data.append(
                {
                    "structural_pattern": struct,
                    "num_simulations": len(sims),
                    "simulation_ids": ",".join([str(s) for s in sorted(sims)]),
                }
            )

        sharing_df = pd.DataFrame(sharing_data)
        sharing_df.to_csv(
            self.report_dir / "simulations_with_similar_patterns.csv", index=False
        )
        print("✓ Created: simulations_with_similar_patterns.csv")

    def create_structural_analysis(self):
        """Analyze structural pattern properties."""
        if self.patterns_df.empty or "structure" not in self.patterns_df.columns:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # 1. Unique elements per pattern
        ax = axes[0, 0]

        def count_unique_elements(structure_str):
            if pd.isna(structure_str):
                return 0
            return len(set(structure_str.split("→")))

        self.patterns_df["unique_elements"] = self.patterns_df["structure"].apply(
            count_unique_elements
        )

        ax.hist(
            self.patterns_df["unique_elements"],
            bins=range(1, 12),
            color="plum",
            edgecolor="black",
            alpha=0.7,
            align="left",
        )
        ax.set_xlabel("Number of Unique Elements in Pattern", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title("Pattern Complexity (Unique Elements)", fontsize=12, weight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # 2. Pattern complexity vs length
        ax = axes[0, 1]
        scatter = ax.scatter(
            self.patterns_df["pattern_length"],
            self.patterns_df["unique_elements"],
            alpha=0.5,
            s=60,
            c="darkgreen",
            edgecolors="black",
            linewidth=0.5,
        )
        ax.set_xlabel("Pattern Length", fontsize=11)
        ax.set_ylabel("Unique Elements", fontsize=11)
        ax.set_title("Pattern Complexity vs Length", fontsize=12, weight="bold")
        ax.grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(
            self.patterns_df["pattern_length"], self.patterns_df["unique_elements"], 1
        )
        p = np.poly1d(z)
        ax.plot(
            self.patterns_df["pattern_length"],
            p(self.patterns_df["pattern_length"]),
            "r--",
            linewidth=2,
            label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}",
        )
        ax.legend()

        # 3. Repetition ratio (length / unique elements)
        ax = axes[1, 0]

        self.patterns_df["repetition_ratio"] = (
            self.patterns_df["pattern_length"] / self.patterns_df["unique_elements"]
        )

        ax.hist(
            self.patterns_df["repetition_ratio"],
            bins=20,
            color="coral",
            edgecolor="black",
            alpha=0.7,
        )
        ax.axvline(
            self.patterns_df["repetition_ratio"].mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {self.patterns_df['repetition_ratio'].mean():.2f}",
        )
        ax.set_xlabel("Repetition Ratio (Length / Unique Elements)", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title("Pattern Repetitiveness", fontsize=12, weight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # 4. Statistics summary
        ax = axes[1, 1]
        ax.axis("off")

        stats_data = [
            ["Total Patterns", len(self.patterns_df)],
            ["Unique Structures", self.patterns_df["structure"].nunique()],
            ["", ""],
            ["Mean Length", f"{self.patterns_df['pattern_length'].mean():.2f}"],
            [
                "Mean Unique Elements",
                f"{self.patterns_df['unique_elements'].mean():.2f}",
            ],
            [
                "Mean Repetition Ratio",
                f"{self.patterns_df['repetition_ratio'].mean():.2f}",
            ],
            ["", ""],
            ["Shortest Pattern", int(self.patterns_df["pattern_length"].min())],
            ["Longest Pattern", int(self.patterns_df["pattern_length"].max())],
            ["", ""],
            ["Min Occurrences", int(self.patterns_df["num_occurrences"].min())],
            ["Max Occurrences", int(self.patterns_df["num_occurrences"].max())],
        ]

        table = ax.table(
            cellText=stats_data,
            colLabels=["Metric", "Value"],
            cellLoc="left",
            colWidths=[0.6, 0.4],
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)

        # Style
        table[(0, 0)].set_facecolor("#9C27B0")
        table[(0, 1)].set_facecolor("#9C27B0")
        table[(0, 0)].set_text_props(weight="bold", color="white")
        table[(0, 1)].set_text_props(weight="bold", color="white")

        ax.set_title(
            "Structural Pattern Statistics", fontsize=13, weight="bold", pad=20
        )

        plt.suptitle("Structural Pattern Analysis", fontsize=16, weight="bold")
        plt.tight_layout()

        plt.savefig(
            self.report_dir / "04_structural_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("✓ Created: 04_structural_analysis.png")

    def create_similarity_matrix(self):
        """Create similarity heatmap for simulations with patterns."""
        if self.patterns_df.empty or "structure" not in self.patterns_df.columns:
            return

        # Create similarity matrix based on structural patterns
        seeds_with_patterns = sorted(self.patterns_df["seed"].unique())

        if len(seeds_with_patterns) < 2:
            print(
                "⚠️  Skipping similarity matrix (need at least 2 simulations with patterns)"
            )
            return

        # Limit to 50 simulations for visualization
        if len(seeds_with_patterns) > 50:
            seeds_with_patterns = seeds_with_patterns[:50]

        n = len(seeds_with_patterns)
        similarity_matrix = np.zeros((n, n))

        seed_to_structure = {}
        for _, row in self.patterns_df.iterrows():
            seed_to_structure[row["seed"]] = row["structure"]

        for i, seed1 in enumerate(seeds_with_patterns):
            for j, seed2 in enumerate(seeds_with_patterns):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Same structure = 1, different = 0
                    if seed_to_structure[seed1] == seed_to_structure[seed2]:
                        similarity_matrix[i, j] = 1.0
                    else:
                        similarity_matrix[i, j] = 0.0

        # Plot
        fig, ax = plt.subplots(figsize=(14, 12))

        im = ax.imshow(similarity_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(
            [f"{s}" for s in seeds_with_patterns], rotation=90, fontsize=8
        )
        ax.set_yticklabels([f"{s}" for s in seeds_with_patterns], fontsize=8)

        ax.set_xlabel("Simulation ID", fontsize=12)
        ax.set_ylabel("Simulation ID", fontsize=12)
        ax.set_title(
            f"Structural Pattern Similarity Matrix\n(First {n} simulations with patterns)",
            fontsize=14,
            weight="bold",
            pad=20,
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Similarity (1=same structure, 0=different)", fontsize=11)

        plt.tight_layout()
        plt.savefig(
            self.report_dir / "05_similarity_matrix.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("✓ Created: 05_similarity_matrix.png")

    def create_pattern_occurrence_analysis(self):
        """Analyze pattern occurrence statistics."""
        if self.patterns_df.empty:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # 1. Occurrences histogram
        ax = axes[0, 0]
        ax.hist(
            self.patterns_df["num_occurrences"],
            bins=30,
            color="indianred",
            edgecolor="black",
            alpha=0.7,
        )
        ax.axvline(
            self.patterns_df["num_occurrences"].mean(),
            color="darkred",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {self.patterns_df['num_occurrences'].mean():.1f}",
        )
        ax.axvline(
            self.patterns_df["num_occurrences"].median(),
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"Median: {self.patterns_df['num_occurrences'].median():.1f}",
        )
        ax.set_xlabel("Number of Occurrences", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title("Pattern Occurrence Distribution", fontsize=12, weight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # 2. Length vs Occurrences
        ax = axes[0, 1]
        scatter = ax.scatter(
            self.patterns_df["pattern_length"],
            self.patterns_df["num_occurrences"],
            alpha=0.5,
            s=50,
            c="purple",
            edgecolors="black",
            linewidth=0.5,
        )
        ax.set_xlabel("Pattern Length", fontsize=11)
        ax.set_ylabel("Number of Occurrences", fontsize=11)
        ax.set_title(
            "Pattern Length vs Repetition Frequency", fontsize=12, weight="bold"
        )
        ax.grid(True, alpha=0.3)

        # Add correlation
        corr = np.corrcoef(
            self.patterns_df["pattern_length"], self.patterns_df["num_occurrences"]
        )[0, 1]
        ax.text(
            0.05,
            0.95,
            f"Correlation: {corr:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # 3. First occurrence position
        ax = axes[1, 0]
        ax.hist(
            self.patterns_df["first_occurrence_step"],
            bins=30,
            color="teal",
            edgecolor="black",
            alpha=0.7,
        )
        ax.set_xlabel(
            "First Occurrence Position (step in active sequence)", fontsize=11
        )
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title("When Do Patterns First Appear?", fontsize=12, weight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # 4. Top 10 most repeated patterns
        ax = axes[1, 1]
        ax.axis("off")

        top_repeated = self.patterns_df.nlargest(10, "num_occurrences")

        table_data = []
        for _, row in top_repeated.iterrows():
            struct = row["structure"] if "structure" in row else "N/A"
            if len(struct) > 30:
                struct = struct[:27] + "..."
            table_data.append(
                [
                    int(row["seed"]),
                    struct,
                    int(row["pattern_length"]),
                    int(row["num_occurrences"]),
                ]
            )

        table = ax.table(
            cellText=table_data,
            colLabels=["Sim ID", "Structure", "Length", "Occurs"],
            cellLoc="center",
            colWidths=[0.15, 0.5, 0.15, 0.2],
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)

        # Style header
        for i in range(4):
            table[(0, i)].set_facecolor("#FF6347")
            table[(0, i)].set_text_props(weight="bold", color="white")

        ax.set_title(
            "Top 10 Most Repeated Patterns", fontsize=12, weight="bold", pad=20
        )

        plt.suptitle("Pattern Occurrence Analysis", fontsize=16, weight="bold")
        plt.tight_layout()

        plt.savefig(
            self.report_dir / "06_occurrence_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("✓ Created: 06_occurrence_analysis.png")

    def save_summary_statistics(self):
        """Save comprehensive summary statistics to CSV."""
        if self.patterns_df.empty:
            return

        summary = {
            "total_simulations": len(self.metadata_df),
            "simulations_with_patterns": self.metadata_df["has_pattern"].sum(),
            "pattern_detection_rate": self.metadata_df["has_pattern"].mean(),
            "total_patterns_detected": len(self.patterns_df),
            "unique_structural_patterns": (
                self.patterns_df["structure"].nunique()
                if "structure" in self.patterns_df.columns
                else 0
            ),
            "mean_pattern_length": self.patterns_df["pattern_length"].mean(),
            "median_pattern_length": self.patterns_df["pattern_length"].median(),
            "std_pattern_length": self.patterns_df["pattern_length"].std(),
            "min_pattern_length": self.patterns_df["pattern_length"].min(),
            "max_pattern_length": self.patterns_df["pattern_length"].max(),
            "mean_occurrences": self.patterns_df["num_occurrences"].mean(),
            "median_occurrences": self.patterns_df["num_occurrences"].median(),
            "max_occurrences": self.patterns_df["num_occurrences"].max(),
        }

        summary_df = pd.DataFrame([summary]).T
        summary_df.columns = ["Value"]
        summary_df.to_csv(self.report_dir / "summary_statistics.csv")
        print("✓ Created: summary_statistics.csv")


def generate_pattern_report(batch_results_dir):
    """
    Generate comprehensive pattern analysis report.

    Args:
        batch_results_dir: Path to batch_simulations_* directory

    Returns:
        Path to report directory
    """
    report = PatternAnalysisReport(batch_results_dir)
    report.generate_full_report()
    return report.report_dir


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        batch_dir = sys.argv[1]
    else:
        # Find most recent batch directory
        import glob

        batch_dirs = glob.glob("batch_simulations_*")
        if not batch_dirs:
            print("❌ No batch simulation directories found!")
            print("Usage: python pattern_report_generator.py <batch_directory>")
            sys.exit(1)
        batch_dir = sorted(batch_dirs)[-1]
        print(f"Using most recent batch: {batch_dir}")

    report_dir = generate_pattern_report(batch_dir)
    print(f"\n🎉 Open the report directory to view all visualizations: {report_dir}")

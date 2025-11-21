"""
Plotting script for simulation results.
Creates 4 visualizations for each simulation:
1. Line plot of 8 behaviors satisfaction levels over time
2. Spider/radar plot of behavior frequency
3. Density curve of behavior frequencies
4. Descriptive statistics table
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats


def plot_satisfaction_timeseries(satisfaction_df, output_path):
    """
    Plot satisfaction levels for all 8 motives over time.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    motives = [col for col in satisfaction_df.columns if col != 'step']

    for motive in motives:
        ax.plot(satisfaction_df['step'], satisfaction_df[motive],
                label=motive, linewidth=2, alpha=0.8)

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Satisfaction Level', fontsize=12)
    ax.set_title('Satisfaction Levels Over Time for 8 Motives',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_behavior_spider(behavior_counts, output_path):
    """
    Create spider/radar plot showing frequency of each behavior.
    """
    categories = list(behavior_counts.keys())
    values = list(behavior_counts.values())

    # Number of variables
    num_vars = len(categories)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Complete the circle
    values += values[:1]
    angles += angles[:1]
    categories += categories[:1]

    # Create plot
    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw=dict(projection='polar'))

    # Plot data
    ax.plot(angles, values, 'o-', linewidth=2, color='#1f77b4')
    ax.fill(angles, values, alpha=0.25, color='#1f77b4')

    # Fix axis to go in the right order
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1], size=12)

    # Set title
    ax.set_title('Behavior Frequency Distribution (Spider Plot)',
                 size=14, fontweight='bold', pad=20)

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_behavior_density(behavior_counts, output_path):
    """
    Create density curve of behavior frequencies.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    counts = list(behavior_counts.values())

    # Create density plot
    if len(counts) > 1 and np.std(counts) > 0:
        # Use KDE for density estimation
        density = stats.gaussian_kde(counts)
        x_range = np.linspace(min(counts), max(counts), 200)
        ax.plot(x_range, density(x_range), linewidth=2, color='#2ca02c')
        ax.fill_between(x_range, density(x_range), alpha=0.3, color='#2ca02c')
    else:
        # If all values are the same, just plot a bar
        ax.bar(counts, [1] * len(counts), width=0.5,
               color='#2ca02c', alpha=0.6)

    # Add histogram overlay
    ax.hist(counts, bins=min(10, len(set(counts))),
            alpha=0.4, color='#d62728', density=True, label='Histogram')

    ax.set_xlabel('Number of Behavior Occurrences', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Density Distribution of Behavior Frequencies',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_descriptive_stats_table(behavior_counts, satisfaction_df,
                                   output_path):
    """
    Create a table with descriptive statistics as a PNG.
    """
    # Calculate statistics
    counts = list(behavior_counts.values())

    stats_dict = {
        'Metric': [
            'Total Behaviors',
            'Mean Frequency',
            'Median Frequency',
            'Std Dev Frequency',
            'Min Frequency',
            'Max Frequency',
            'Most Common Behavior',
            'Total Time Steps'
        ],
        'Value': [
            sum(counts),
            f"{np.mean(counts):.2f}",
            f"{np.median(counts):.2f}",
            f"{np.std(counts):.2f}",
            min(counts),
            max(counts),
            max(behavior_counts, key=behavior_counts.get),
            len(satisfaction_df)
        ]
    }

    stats_df = pd.DataFrame(stats_dict)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=stats_df.values,
                     colLabels=stats_df.columns,
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.6, 0.4])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Header styling
    for i in range(len(stats_df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(stats_df) + 1):
        for j in range(len(stats_df.columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#FFFFFF')

    plt.title('Descriptive Statistics', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def count_behaviors(behavior_df):
    """
    Count occurrences of each behavior from behavior sequence.
    """
    # Filter out empty rows
    behaviors = behavior_df[
        behavior_df['active_behavior'].notna() &
        (behavior_df['active_behavior'] != '')
    ]

    # Count each behavior
    behavior_counts = behaviors['active_behavior'].value_counts().to_dict()

    # Ensure all 8 motives are present (even if count is 0)
    all_motives = [f'motive_{i}' for i in range(1, 9)]
    for motive in all_motives:
        if motive not in behavior_counts:
            behavior_counts[motive] = 0

    # Sort by motive number
    behavior_counts = dict(sorted(behavior_counts.items()))

    return behavior_counts


def process_simulation(sim_path):
    """
    Process a single simulation and create all plots.
    """
    sim_path = Path(sim_path)

    # Read data
    behavior_df = pd.read_csv(sim_path / 'behavior_sequence.csv')
    satisfaction_df = pd.read_csv(sim_path / 'satisfaction_timeseries.csv')

    # Count behaviors
    behavior_counts = count_behaviors(behavior_df)

    # Create output directory for plots
    plots_dir = sim_path / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Generate all plots
    print("  Creating satisfaction timeseries plot...")
    plot_satisfaction_timeseries(
        satisfaction_df,
        plots_dir / '1_satisfaction_timeseries.png')

    print("  Creating spider plot...")
    plot_behavior_spider(
        behavior_counts,
        plots_dir / '2_behavior_spider.png')

    print("  Creating density plot...")
    plot_behavior_density(
        behavior_counts,
        plots_dir / '3_behavior_density.png')

    print("  Creating statistics table...")
    create_descriptive_stats_table(
        behavior_counts, satisfaction_df,
        plots_dir / '4_descriptive_stats.png')

    print(f"  [OK] All plots saved to: {plots_dir}")


def process_batch(batch_path):
    """
    Process all simulations in a batch.
    """
    batch_path = Path(batch_path)

    if not batch_path.exists():
        print(f"Error: Batch path does not exist: {batch_path}")
        return

    # Find all simulation directories
    sim_dirs = sorted([d for d in batch_path.iterdir()
                      if d.is_dir() and d.name.startswith('simulation_')])

    if not sim_dirs:
        print(f"No simulation directories found in {batch_path}")
        return

    print(f"Found {len(sim_dirs)} simulations in batch: {batch_path.name}")
    print("=" * 60)

    for sim_dir in sim_dirs:
        print(f"\nProcessing {sim_dir.name}...")
        try:
            process_simulation(sim_dir)
        except Exception as e:
            print(f"  [ERROR] Error processing {sim_dir.name}: {e}")

    print("\n" + "=" * 60)
    print("Batch processing complete!")


if __name__ == "__main__":
    import sys

    # Default to latest batch if no argument provided
    if len(sys.argv) > 1:
        batch_path = sys.argv[1]
    else:
        # Find the most recent batch in data directory
        data_dir = Path(__file__).parent.parent / 'data'
        batches = sorted([d for d in data_dir.iterdir()
                         if d.is_dir() and d.name.startswith('my_batch_')])

        if not batches:
            print("No batch directories found in data folder!")
            sys.exit(1)

        batch_path = batches[-1]  # Most recent
        print(f"No batch specified, using most recent: {batch_path.name}")

    process_batch(batch_path)

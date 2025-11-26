"""
Batch-level Aggregate Analysis Script

Combines data from all simulations in a batch and performs analyses on the
aggregated dataset. This provides an overall picture of behavior patterns
across all simulations in the batch.

Creates:
- Aggregated Markov chain (transition matrix from all simulations combined)
- Aggregated plots (satisfaction timeseries, behavior distribution, etc.)
- Statistical summaries across the entire batch
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from scipy import stats


def aggregate_behavior_sequences(batch_path):
    """
    Combine all behavior sequences from all simulations in the batch.
    """
    batch_path = Path(batch_path)
    sim_dirs = sorted([d for d in batch_path.iterdir()
                      if d.is_dir() and d.name.startswith('simulation_')])

    all_behaviors = []

    for sim_dir in sim_dirs:
        behavior_file = sim_dir / 'behavior_sequence.csv'
        if behavior_file.exists():
            df = pd.read_csv(behavior_file)
            behaviors = df[
                df['active_behavior'].notna() &
                (df['active_behavior'] != '')
            ]['active_behavior'].tolist()
            all_behaviors.extend(behaviors)

    return all_behaviors


def aggregate_satisfaction_data(batch_path):
    """
    Combine all satisfaction timeseries from all simulations.
    Returns a list of DataFrames (one per simulation).
    """
    batch_path = Path(batch_path)
    sim_dirs = sorted([d for d in batch_path.iterdir()
                      if d.is_dir() and d.name.startswith('simulation_')])

    all_satisfaction_dfs = []

    for sim_dir in sim_dirs:
        satisfaction_file = sim_dir / 'satisfaction_timeseries.csv'
        if satisfaction_file.exists():
            df = pd.read_csv(satisfaction_file)
            all_satisfaction_dfs.append(df)

    return all_satisfaction_dfs


def calculate_aggregate_transition_matrix(all_behaviors):
    """
    Calculate transition matrix from aggregated behavior sequences.
    """
    all_motives = [f'motive_{i}' for i in range(1, 9)]
    n_motives = len(all_motives)

    # Initialize transition count matrix
    transition_counts = pd.DataFrame(
        0,
        index=all_motives,
        columns=all_motives
    )

    # Count transitions
    for i in range(len(all_behaviors) - 1):
        current_behavior = all_behaviors[i]
        next_behavior = all_behaviors[i + 1]
        if current_behavior in all_motives and next_behavior in all_motives:
            transition_counts.loc[current_behavior, next_behavior] += 1

    # Convert counts to probabilities
    transition_matrix = transition_counts.astype(float).copy()
    for motive in all_motives:
        row_sum = transition_counts.loc[motive].sum()
        if row_sum > 0:
            transition_matrix.loc[motive] = transition_counts.loc[motive] / row_sum
        else:
            # If no transitions from this motive, use uniform distribution
            transition_matrix.loc[motive] = 1.0 / n_motives

    return transition_matrix, transition_counts


def visualize_aggregate_markov_chain(transition_matrix, output_path, threshold=0.05):
    """
    Create Markov chain visualization for aggregated data.
    """
    G = nx.DiGraph()
    motives = transition_matrix.index.tolist()
    G.add_nodes_from(motives)

    # Calculate statistics for non-self-loop transitions
    non_loop_probs = []
    for from_motive in motives:
        for to_motive in motives:
            if from_motive != to_motive:
                prob = transition_matrix.loc[from_motive, to_motive]
                non_loop_probs.append(prob)

    mean_prob = np.mean(non_loop_probs)
    std_prob = np.std(non_loop_probs)
    high_threshold = mean_prob + 1.5 * std_prob

    # Add edges (excluding self-loops)
    for from_motive in motives:
        for to_motive in motives:
            if from_motive != to_motive:
                prob = transition_matrix.loc[from_motive, to_motive]
                if prob > threshold:
                    is_high = prob > high_threshold
                    G.add_edge(from_motive, to_motive, weight=prob, is_high=is_high)

    # Set up plot
    fig, ax = plt.subplots(figsize=(14, 14))
    pos = nx.circular_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color='lightblue',
        node_size=3000,
        alpha=0.9,
        ax=ax
    )

    nx.draw_networkx_labels(
        G, pos,
        font_size=12,
        font_weight='bold',
        ax=ax
    )

    # Separate edges
    normal_edges = [(u, v) for u, v in G.edges() if not G[u][v]['is_high']]
    high_edges = [(u, v) for u, v in G.edges() if G[u][v]['is_high']]

    all_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(all_weights) if all_weights else 1

    # Draw normal edges
    if normal_edges:
        normal_widths = [3 + (G[u][v]['weight'] / max_weight) * 7 for u, v in normal_edges]
        nx.draw_networkx_edges(
            G, pos,
            edgelist=normal_edges,
            width=normal_widths,
            alpha=0.6,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1',
            ax=ax
        )

    # Draw high-probability edges
    if high_edges:
        high_widths = [3 + (G[u][v]['weight'] / max_weight) * 7 for u, v in high_edges]
        nx.draw_networkx_edges(
            G, pos,
            edgelist=high_edges,
            width=high_widths,
            alpha=0.8,
            edge_color='red',
            arrows=True,
            arrowsize=25,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1',
            ax=ax
        )

    # Draw edge labels
    normal_edge_labels = {(u, v): f'{G[u][v]["weight"]:.2f}' for u, v in normal_edges}
    if normal_edge_labels:
        nx.draw_networkx_edge_labels(
            G, pos,
            normal_edge_labels,
            font_size=9,
            font_color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
            ax=ax
        )

    high_edge_labels = {(u, v): f'{G[u][v]["weight"]:.2f}' for u, v in high_edges}
    if high_edge_labels:
        nx.draw_networkx_edge_labels(
            G, pos,
            high_edge_labels,
            font_size=10,
            font_color='red',
            font_weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
            ax=ax
        )

    ax.set_title(
        f'Aggregate Markov Chain: All Simulations Combined\n'
        f'Red = High probability (> {high_threshold:.2f}), Gray = Normal\n'
        f'Mean: {mean_prob:.3f}, Std: {std_prob:.3f}',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_aggregate_heatmap(transition_matrix, output_path):
    """
    Create heatmap for aggregated transition matrix.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    heatmap_data = transition_matrix.copy()
    np.fill_diagonal(heatmap_data.values, np.nan)

    im = ax.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    motives = transition_matrix.index.tolist()
    ax.set_xticks(np.arange(len(motives)))
    ax.set_yticks(np.arange(len(motives)))
    ax.set_xticklabels(motives, rotation=45, ha='right')
    ax.set_yticklabels(motives)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Transition Probability', rotation=270, labelpad=20)

    for i in range(len(motives)):
        for j in range(len(motives)):
            if i != j:
                value = transition_matrix.iloc[i, j]
                ax.text(j, i, f'{value:.2f}',
                       ha='center', va='center',
                       color='white' if value > 0.5 else 'black',
                       fontsize=9)
            else:
                ax.text(j, i, 'X',
                       ha='center', va='center',
                       color='gray',
                       fontsize=12,
                       fontweight='bold')

    ax.set_xlabel('To Behavior', fontsize=12, fontweight='bold')
    ax.set_ylabel('From Behavior', fontsize=12, fontweight='bold')
    ax.set_title('Aggregate Transition Matrix (All Simulations)',
                fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_aggregate_satisfaction_timeseries(satisfaction_dfs, output_path):
    """
    Plot average satisfaction across all simulations.
    """
    # Calculate mean and std across all simulations
    # First, ensure all have the same length
    min_length = min(len(df) for df in satisfaction_dfs)

    # Truncate all to same length
    truncated_dfs = [df.head(min_length) for df in satisfaction_dfs]

    # Stack and calculate statistics
    all_motives = [f'motive_{i}' for i in range(1, 9)]

    fig, ax = plt.subplots(figsize=(12, 6))

    for motive in all_motives:
        # Get all values for this motive across all simulations
        motive_data = np.array([df[motive].values for df in truncated_dfs])
        mean_values = np.mean(motive_data, axis=0)
        std_values = np.std(motive_data, axis=0)
        steps = truncated_dfs[0]['step'].values

        # Plot mean line
        ax.plot(steps, mean_values, label=motive, linewidth=2, alpha=0.8)
        # Add shaded std region
        ax.fill_between(steps,
                        mean_values - std_values,
                        mean_values + std_values,
                        alpha=0.2)

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Mean Satisfaction Level', fontsize=12)
    ax.set_title(f'Aggregate Satisfaction Over Time (n={len(satisfaction_dfs)} simulations)\n'
                 'Lines = Mean, Shaded = ±1 SD',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_aggregate_behavior_distribution(all_behaviors, output_path):
    """
    Plot distribution of behaviors across all simulations.
    """
    all_motives = [f'motive_{i}' for i in range(1, 9)]
    behavior_counts = {motive: all_behaviors.count(motive) for motive in all_motives}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart
    motives = list(behavior_counts.keys())
    counts = list(behavior_counts.values())

    ax1.bar(motives, counts, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Behavior', fontsize=12)
    ax1.set_ylabel('Total Frequency', fontsize=12)
    ax1.set_title('Aggregate Behavior Frequencies', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add count labels on bars
    for i, (motive, count) in enumerate(behavior_counts.items()):
        ax1.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')

    # Pie chart
    ax2.pie(counts, labels=motives, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Behavior Distribution Proportions', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_aggregate_statistics_summary(all_behaviors, transition_counts, output_path):
    """
    Create summary statistics table for the aggregated data.
    """
    all_motives = [f'motive_{i}' for i in range(1, 9)]
    behavior_counts = {motive: all_behaviors.count(motive) for motive in all_motives}
    counts = list(behavior_counts.values())

    stats_data = {
        'Metric': [
            'Total Behaviors',
            'Mean Frequency',
            'Median Frequency',
            'Std Dev',
            'Min Frequency',
            'Max Frequency',
            'Range',
            'Coefficient of Variation',
            'Gini Coefficient',
            'Shannon Entropy'
        ],
        'Value': [
            sum(counts),
            np.mean(counts),
            np.median(counts),
            np.std(counts, ddof=1),
            np.min(counts),
            np.max(counts),
            np.max(counts) - np.min(counts),
            np.std(counts, ddof=1) / np.mean(counts) if np.mean(counts) > 0 else 0,
            calculate_gini(counts),
            calculate_entropy(counts)
        ]
    }

    df = pd.DataFrame(stats_data)
    df['Value'] = df['Value'].round(3)

    # Save as CSV
    df.to_csv(output_path.parent / 'aggregate_statistics.csv', index=False)

    # Create table image
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='left',
        loc='center',
        colWidths=[0.6, 0.4]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.title('Aggregate Statistics Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def calculate_gini(counts):
    """Calculate Gini coefficient."""
    counts_array = np.array(counts)
    if np.sum(counts_array) == 0:
        return 0
    sorted_counts = np.sort(counts_array)
    n = len(sorted_counts)
    cumsum = np.cumsum(sorted_counts)
    return (2 * np.sum((np.arange(1, n + 1)) * sorted_counts)) / (
        n * np.sum(sorted_counts)
    ) - (n + 1) / n


def calculate_entropy(counts):
    """Calculate Shannon entropy."""
    counts_array = np.array(counts)
    if np.sum(counts_array) == 0:
        return 0
    probabilities = counts_array / np.sum(counts_array)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))


def process_batch(batch_path):
    """
    Process entire batch and create aggregate analyses.
    """
    batch_path = Path(batch_path)

    if not batch_path.exists():
        print(f"Error: Batch path does not exist: {batch_path}")
        return

    print(f"Processing batch: {batch_path.name}")
    print("=" * 60)

    # Create aggregate output directory
    aggregate_dir = batch_path / 'aggregate_analysis'
    aggregate_dir.mkdir(exist_ok=True)

    print("\n1. Aggregating behavior sequences from all simulations...")
    all_behaviors = aggregate_behavior_sequences(batch_path)
    print(f"   Total behaviors collected: {len(all_behaviors)}")

    print("\n2. Aggregating satisfaction data from all simulations...")
    satisfaction_dfs = aggregate_satisfaction_data(batch_path)
    print(f"   Total simulations with satisfaction data: {len(satisfaction_dfs)}")

    print("\n3. Calculating aggregate transition matrix...")
    transition_matrix, transition_counts = calculate_aggregate_transition_matrix(all_behaviors)
    transition_matrix.to_csv(aggregate_dir / 'aggregate_transition_matrix.csv')
    transition_counts.to_csv(aggregate_dir / 'aggregate_transition_counts.csv')
    print("   Saved: aggregate_transition_matrix.csv")
    print("   Saved: aggregate_transition_counts.csv")

    print("\n4. Creating aggregate Markov chain visualization...")
    visualize_aggregate_markov_chain(
        transition_matrix,
        aggregate_dir / 'aggregate_markov_chain_network.png'
    )
    print("   Saved: aggregate_markov_chain_network.png")

    print("\n5. Creating aggregate transition heatmap...")
    create_aggregate_heatmap(
        transition_matrix,
        aggregate_dir / 'aggregate_transition_heatmap.png'
    )
    print("   Saved: aggregate_transition_heatmap.png")

    print("\n6. Creating aggregate satisfaction timeseries plot...")
    plot_aggregate_satisfaction_timeseries(
        satisfaction_dfs,
        aggregate_dir / 'aggregate_satisfaction_timeseries.png'
    )
    print("   Saved: aggregate_satisfaction_timeseries.png")

    print("\n7. Creating aggregate behavior distribution plots...")
    plot_aggregate_behavior_distribution(
        all_behaviors,
        aggregate_dir / 'aggregate_behavior_distribution.png'
    )
    print("   Saved: aggregate_behavior_distribution.png")

    print("\n8. Creating aggregate statistics summary...")
    create_aggregate_statistics_summary(
        all_behaviors,
        transition_counts,
        aggregate_dir / 'aggregate_statistics_table.png'
    )
    print("   Saved: aggregate_statistics_table.png")
    print("   Saved: aggregate_statistics.csv")

    print("\n" + "=" * 60)
    print(f"[OK] Aggregate analysis complete!")
    print(f"     Results saved to: {aggregate_dir}")


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
        print(f"No batch specified, using most recent: {batch_path.name}\n")

    process_batch(batch_path)

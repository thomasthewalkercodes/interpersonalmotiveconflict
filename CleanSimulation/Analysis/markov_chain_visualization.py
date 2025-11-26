"""
Markov Chain Visualization Script

Analyzes behavior sequences to create transition probability matrices
and generates visual Markov chain diagrams showing how behaviors transition
from one to another.

The visualization shows:
- Nodes: Each of the 8 motives
- Edges: Transition probabilities (arrows showing direction)
- Edge thickness: Proportional to transition probability
- Self-loops: Probability of staying in the same behavior
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path


def calculate_transition_matrix(behavior_df):
    """
    Calculate transition probability matrix from behavior sequence.

    Returns a matrix where entry (i,j) represents the probability
    of transitioning from behavior i to behavior j.
    """
    # Filter out empty rows
    behaviors = behavior_df[
        behavior_df['active_behavior'].notna() &
        (behavior_df['active_behavior'] != '')
    ]['active_behavior'].tolist()

    # Define all motives
    all_motives = [f'motive_{i}' for i in range(1, 9)]
    n_motives = len(all_motives)

    # Initialize transition count matrix
    transition_counts = pd.DataFrame(
        0,
        index=all_motives,
        columns=all_motives
    )

    # Count transitions
    for i in range(len(behaviors) - 1):
        current_behavior = behaviors[i]
        next_behavior = behaviors[i + 1]
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

    return transition_matrix


def visualize_markov_chain(transition_matrix, output_path, threshold=0.05):
    """
    Create a visual representation of the Markov chain.

    Args:
        transition_matrix: DataFrame with transition probabilities
        output_path: Where to save the PNG
        threshold: Minimum probability to display an edge (default 0.05)
    """
    # Create directed graph
    G = nx.DiGraph()

    # Add nodes
    motives = transition_matrix.index.tolist()
    G.add_nodes_from(motives)

    # Calculate statistics for non-self-loop transitions
    non_loop_probs = []
    for i, from_motive in enumerate(motives):
        for j, to_motive in enumerate(motives):
            if from_motive != to_motive:  # Exclude self-loops
                prob = transition_matrix.loc[from_motive, to_motive]
                non_loop_probs.append(prob)

    # Calculate mean and std for highlighting
    mean_prob = np.mean(non_loop_probs)
    std_prob = np.std(non_loop_probs)
    high_threshold = mean_prob + 1.5 * std_prob  # Significantly higher threshold

    # Add edges with weights (excluding self-loops, only if probability > threshold)
    for i, from_motive in enumerate(motives):
        for j, to_motive in enumerate(motives):
            if from_motive != to_motive:  # Exclude self-loops
                prob = transition_matrix.loc[from_motive, to_motive]
                if prob > threshold:
                    # Mark if this is a high probability transition
                    is_high = prob > high_threshold
                    G.add_edge(from_motive, to_motive, weight=prob, is_high=is_high)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 14))

    # Use circular layout for the 8 motives
    pos = nx.circular_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color='lightblue',
        node_size=3000,
        alpha=0.9,
        ax=ax
    )

    # Draw node labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=12,
        font_weight='bold',
        ax=ax
    )

    # Separate edges into normal and high-probability
    normal_edges = [(u, v) for u, v in G.edges() if not G[u][v]['is_high']]
    high_edges = [(u, v) for u, v in G.edges() if G[u][v]['is_high']]

    # Get weights for edge widths
    all_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(all_weights) if all_weights else 1

    # Draw normal edges (gray)
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

    # Draw high-probability edges (red/highlighted)
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

    # Draw edge labels (probabilities) with color coding
    # Normal edge labels (black)
    normal_edge_labels = {(u, v): f'{G[u][v]["weight"]:.2f}'
                          for u, v in normal_edges}
    if normal_edge_labels:
        nx.draw_networkx_edge_labels(
            G, pos,
            normal_edge_labels,
            font_size=9,
            font_color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
            ax=ax
        )

    # High probability edge labels (red)
    high_edge_labels = {(u, v): f'{G[u][v]["weight"]:.2f}'
                        for u, v in high_edges}
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
        f'Markov Chain: Behavior Transition Probabilities (Self-loops excluded)\n'
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


def create_transition_heatmap(transition_matrix, output_path):
    """
    Create a heatmap visualization of the transition matrix (excluding self-loops).
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a copy and set diagonal to NaN to visually distinguish self-loops
    heatmap_data = transition_matrix.copy()
    np.fill_diagonal(heatmap_data.values, np.nan)

    # Create heatmap
    im = ax.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    # Set ticks and labels
    motives = transition_matrix.index.tolist()
    ax.set_xticks(np.arange(len(motives)))
    ax.set_yticks(np.arange(len(motives)))
    ax.set_xticklabels(motives, rotation=45, ha='right')
    ax.set_yticklabels(motives)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Transition Probability', rotation=270, labelpad=20)

    # Add text annotations (skip diagonal/self-loops)
    for i in range(len(motives)):
        for j in range(len(motives)):
            if i != j:  # Skip self-loops
                value = transition_matrix.iloc[i, j]
                text = ax.text(j, i, f'{value:.2f}',
                              ha='center', va='center',
                              color='white' if value > 0.5 else 'black',
                              fontsize=9)
            else:
                # Mark self-loops as excluded
                text = ax.text(j, i, 'X',
                              ha='center', va='center',
                              color='gray',
                              fontsize=12,
                              fontweight='bold')

    ax.set_xlabel('To Behavior', fontsize=12, fontweight='bold')
    ax.set_ylabel('From Behavior', fontsize=12, fontweight='bold')
    ax.set_title('Transition Probability Matrix (Self-loops excluded)',
                fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def process_simulation(sim_path):
    """
    Process a single simulation and create Markov chain visualizations.
    """
    sim_path = Path(sim_path)

    # Read behavior data
    behavior_df = pd.read_csv(sim_path / 'behavior_sequence.csv')

    # Calculate transition matrix
    print("  Calculating transition probabilities...")
    transition_matrix = calculate_transition_matrix(behavior_df)

    # Create output directory
    markov_dir = sim_path / 'markov_chain'
    markov_dir.mkdir(exist_ok=True)

    # Save transition matrix as CSV
    transition_matrix.to_csv(markov_dir / 'transition_matrix.csv')
    print(f"    Saved: transition_matrix.csv")

    # Create network diagram
    print("  Creating Markov chain network diagram...")
    visualize_markov_chain(
        transition_matrix,
        markov_dir / 'markov_chain_network.png',
        threshold=0.05
    )
    print(f"    Saved: markov_chain_network.png")

    # Create heatmap
    print("  Creating transition matrix heatmap...")
    create_transition_heatmap(
        transition_matrix,
        markov_dir / 'transition_matrix_heatmap.png'
    )
    print(f"    Saved: transition_matrix_heatmap.png")

    print(f"  [OK] Markov chain analysis saved to: {markov_dir}")


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
    print("Markov chain analysis complete!")


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

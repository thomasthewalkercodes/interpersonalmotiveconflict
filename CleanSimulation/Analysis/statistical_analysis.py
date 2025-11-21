"""
Statistical analysis script for simulation behavior frequency distributions.

Performs various statistical tests on behavior frequencies:
- Chi-square goodness of fit test (uniform distribution)
- Normality tests (Shapiro-Wilk, Anderson-Darling)
- Descriptive statistics
- Entropy measures
- Gini coefficient (inequality measure)
- Frequency distribution metrics

Results are saved as CSV files in each simulation folder.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import entropy


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


def calculate_gini_coefficient(counts):
    """
    Calculate Gini coefficient as a measure of inequality in distribution.
    0 = perfect equality, 1 = perfect inequality
    """
    counts_array = np.array(list(counts.values()))
    if np.sum(counts_array) == 0:
        return 0

    # Sort the array
    sorted_counts = np.sort(counts_array)
    n = len(sorted_counts)
    cumsum = np.cumsum(sorted_counts)
    return (2 * np.sum((np.arange(1, n + 1)) * sorted_counts)) / (
        n * np.sum(sorted_counts)
    ) - (n + 1) / n


def calculate_shannon_entropy(counts):
    """
    Calculate Shannon entropy as a measure of diversity/randomness.
    Higher values indicate more uniform distribution.
    """
    counts_array = np.array(list(counts.values()))
    if np.sum(counts_array) == 0:
        return 0

    # Normalize to probabilities
    probabilities = counts_array / np.sum(counts_array)
    return entropy(probabilities, base=2)


def test_uniformity(behavior_counts):
    """
    Chi-square goodness of fit test for uniform distribution.
    Tests whether behavior frequencies are uniformly distributed.
    """
    observed = np.array(list(behavior_counts.values()))
    n_behaviors = len(observed)
    total = np.sum(observed)

    # Expected frequencies under uniform distribution
    expected = np.full(n_behaviors, total / n_behaviors)

    # Perform chi-square test
    chi2_stat, p_value = stats.chisquare(observed, expected)

    return {
        'test': 'Chi-Square Uniformity Test',
        'statistic': chi2_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'interpretation': (
            'Reject uniform distribution' if p_value < 0.05
            else 'Cannot reject uniform distribution'
        )
    }


def test_normality(behavior_counts):
    """
    Test whether behavior frequencies follow a normal distribution.
    """
    counts_array = np.array(list(behavior_counts.values()))

    results = []

    # Shapiro-Wilk test
    if len(counts_array) >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(counts_array)
        results.append({
            'test': 'Shapiro-Wilk Normality Test',
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'significant': shapiro_p < 0.05,
            'interpretation': (
                'Reject normality' if shapiro_p < 0.05
                else 'Cannot reject normality'
            )
        })

    # Anderson-Darling test
    anderson_result = stats.anderson(counts_array, dist='norm')
    # Use 5% significance level (index 2)
    results.append({
        'test': 'Anderson-Darling Normality Test',
        'statistic': anderson_result.statistic,
        'critical_value_5%': anderson_result.critical_values[2],
        'significant': (
            anderson_result.statistic > anderson_result.critical_values[2]
        ),
        'interpretation': (
            'Reject normality'
            if anderson_result.statistic > anderson_result.critical_values[2]
            else 'Cannot reject normality'
        )
    })

    return results


def calculate_descriptive_stats(behavior_counts):
    """
    Calculate comprehensive descriptive statistics.
    """
    counts_array = np.array(list(behavior_counts.values()))

    return {
        'total_behaviors': int(np.sum(counts_array)),
        'mean_frequency': float(np.mean(counts_array)),
        'median_frequency': float(np.median(counts_array)),
        'std_dev': float(np.std(counts_array, ddof=1)),
        'variance': float(np.var(counts_array, ddof=1)),
        'min_frequency': int(np.min(counts_array)),
        'max_frequency': int(np.max(counts_array)),
        'range': int(np.max(counts_array) - np.min(counts_array)),
        'cv': (
            float(np.std(counts_array, ddof=1) / np.mean(counts_array))
            if np.mean(counts_array) > 0 else 0
        ),
        'skewness': float(stats.skew(counts_array)),
        'kurtosis': float(stats.kurtosis(counts_array))
    }


def analyze_behavior_frequencies(behavior_counts):
    """
    Perform comprehensive statistical analysis on behavior frequencies.
    """
    # Descriptive statistics
    descriptive = calculate_descriptive_stats(behavior_counts)

    # Distribution measures
    gini = calculate_gini_coefficient(behavior_counts)
    shannon_ent = calculate_shannon_entropy(behavior_counts)
    max_entropy = np.log2(len(behavior_counts))  # Maximum possible entropy

    distribution_measures = {
        'gini_coefficient': float(gini),
        'shannon_entropy': float(shannon_ent),
        'max_entropy': float(max_entropy),
        'normalized_entropy': float(shannon_ent / max_entropy) if max_entropy > 0 else 0
    }

    # Hypothesis tests
    uniformity_test = test_uniformity(behavior_counts)
    normality_tests = test_normality(behavior_counts)

    return {
        'descriptive': descriptive,
        'distribution_measures': distribution_measures,
        'uniformity_test': uniformity_test,
        'normality_tests': normality_tests
    }


def save_analysis_results(results, output_dir):
    """
    Save analysis results as CSV files in the specified directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 1. Descriptive statistics CSV
    descriptive_df = pd.DataFrame([results['descriptive']])
    descriptive_df.to_csv(
        output_dir / 'descriptive_statistics.csv',
        index=False
    )

    # 2. Distribution measures CSV
    distribution_df = pd.DataFrame([results['distribution_measures']])
    distribution_df.to_csv(
        output_dir / 'distribution_measures.csv',
        index=False
    )

    # 3. Hypothesis tests CSV
    tests_data = []

    # Add uniformity test
    tests_data.append({
        'test_name': results['uniformity_test']['test'],
        'statistic': results['uniformity_test']['statistic'],
        'p_value': results['uniformity_test'].get('p_value', np.nan),
        'critical_value': np.nan,
        'significant': results['uniformity_test']['significant'],
        'interpretation': results['uniformity_test']['interpretation']
    })

    # Add normality tests
    for test in results['normality_tests']:
        tests_data.append({
            'test_name': test['test'],
            'statistic': test['statistic'],
            'p_value': test.get('p_value', np.nan),
            'critical_value': test.get('critical_value_5%', np.nan),
            'significant': test['significant'],
            'interpretation': test['interpretation']
        })

    tests_df = pd.DataFrame(tests_data)
    tests_df.to_csv(
        output_dir / 'hypothesis_tests.csv',
        index=False
    )

    print(f"    Saved: descriptive_statistics.csv")
    print(f"    Saved: distribution_measures.csv")
    print(f"    Saved: hypothesis_tests.csv")


def process_simulation(sim_path):
    """
    Process a single simulation and perform statistical analysis.
    """
    sim_path = Path(sim_path)

    # Read behavior data
    behavior_df = pd.read_csv(sim_path / 'behavior_sequence.csv')

    # Count behaviors
    behavior_counts = count_behaviors(behavior_df)

    # Perform statistical analysis
    results = analyze_behavior_frequencies(behavior_counts)

    # Create output directory for analysis results
    analysis_dir = sim_path / 'statistical_analysis'
    analysis_dir.mkdir(exist_ok=True)

    # Save results
    save_analysis_results(results, analysis_dir)

    print(f"  [OK] Statistical analysis saved to: {analysis_dir}")

    return results


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
    print("Statistical analysis complete!")


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

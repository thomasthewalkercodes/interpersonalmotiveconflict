"""
Pattern Analysis Module for Motive Game Engine
Analyzes behavior sequences with STRUCTURE-BASED pattern recognition.

Key insight: [motive_1, motive_2, motive_3] has the same STRUCTURE as
[motive_5, motive_7, motive_2] - both are patterns A→B→C.
"""

import numpy as np
from collections import Counter
import time


class StructuralPatternDetector:
    """
    Detects patterns based on STRUCTURE, not specific motive identities.

    Example:
        Sequence 1: [motive_1, motive_2, motive_1] = [A, B, A] → pattern "ABA"
        Sequence 2: [motive_5, motive_3, motive_5] = [A, B, A] → pattern "ABA"
        Both have the SAME structural pattern!
    """

    @staticmethod
    def encode_structure(sequence):
        """
        Convert a sequence to its structural representation.

        Args:
            sequence: List of behaviors (e.g., ['motive_1', 'motive_2', 'motive_1'])

        Returns:
            tuple: Structural pattern (e.g., (0, 1, 0) for "ABA" pattern)
            dict: Mapping of structure to actual motives

        Example:
            sequence = ['motive_3', 'motive_5', 'motive_3', 'motive_5']
            returns: (0, 1, 0, 1), {0: 'motive_3', 1: 'motive_5'}
        """
        if not sequence:
            return tuple(), {}

        # Create mapping: first unique element = 0, second = 1, etc.
        motive_to_id = {}
        id_to_motive = {}
        next_id = 0

        structure = []
        for motive in sequence:
            if motive not in motive_to_id:
                motive_to_id[motive] = next_id
                id_to_motive[next_id] = motive
                next_id += 1

            structure.append(motive_to_id[motive])

        return tuple(structure), id_to_motive

    @staticmethod
    def find_structural_patterns(sequence, max_pattern_length=50, timeout_seconds=5):
        """
        Find repeating patterns based on STRUCTURE only.

        Args:
            sequence: List of behaviors (with None values removed)
            max_pattern_length: Maximum pattern length
            timeout_seconds: Timeout for search

        Returns:
            dict with pattern info including structural representation
        """
        if not sequence or len(sequence) < 4:
            return None

        start_time = time.time()

        # Limit for performance
        if len(sequence) > 1000:
            sequence = sequence[:1000]

        best_pattern = None
        max_length = 0

        max_possible = min(len(sequence) // 2, max_pattern_length)

        # Try different pattern lengths
        for pattern_len in range(2, max_possible + 1):
            if time.time() - start_time > timeout_seconds:
                break

            # Track structural patterns using dictionary
            structure_to_sequences = {}

            for i in range(len(sequence) - pattern_len + 1):
                subsequence = sequence[i : i + pattern_len]

                # Get structural representation
                structure, mapping = StructuralPatternDetector.encode_structure(
                    subsequence
                )

                if structure not in structure_to_sequences:
                    structure_to_sequences[structure] = {
                        "positions": [i],
                        "examples": [subsequence],
                    }
                else:
                    structure_to_sequences[structure]["positions"].append(i)
                    structure_to_sequences[structure]["examples"].append(subsequence)

            # Find structures that repeat at least twice
            for structure, data in structure_to_sequences.items():
                if len(data["positions"]) >= 2 and pattern_len > max_length:
                    max_length = pattern_len

                    # Use first example as canonical representation
                    canonical_example = data["examples"][0]

                    best_pattern = {
                        "pattern": canonical_example,  # Actual motive sequence
                        "structure": structure,  # Abstract structure (0,1,0,1,...)
                        "structure_string": "→".join(
                            [chr(65 + x) for x in structure[:26]]
                        ),  # A→B→A→B
                        "length": pattern_len,
                        "first_occurrence": data["positions"][0],
                        "num_occurrences": len(data["positions"]),
                        "occurrence_positions": data["positions"][:10],
                        "all_examples": data["examples"][
                            :5
                        ],  # Show first 5 concrete examples
                    }

            # Early stopping
            if max_length > 20:
                break

        return best_pattern

    @staticmethod
    def calculate_structural_similarity(seq1, seq2):
        """
        Calculate similarity based on structural patterns.

        Two sequences are similar if they have similar structural patterns,
        regardless of which specific motives are involved.

        Args:
            seq1, seq2: Lists of behaviors (no None)

        Returns:
            float: Structural similarity (0-1)
        """
        if not seq1 or not seq2:
            return 0.0

        # Encode structures
        struct1, _ = StructuralPatternDetector.encode_structure(seq1)
        struct2, _ = StructuralPatternDetector.encode_structure(seq2)

        # Compare structures using longest common subsequence
        min_len = min(len(struct1), len(struct2))
        max_len = max(len(struct1), len(struct2))

        if max_len == 0:
            return 0.0

        # Count matching positions in structural representation
        matches = sum(1 for i in range(min_len) if struct1[i] == struct2[i])

        return matches / max_len


class PatternDetector:
    """
    Main pattern detector with both specific and structural pattern recognition.
    """

    @staticmethod
    def find_longest_repeating_pattern(
        sequence, max_pattern_length=50, timeout_seconds=5, structural=False
    ):
        """
        Find the longest repeating pattern in ACTIVE behaviors only (skips None).

        Args:
            sequence: List of behaviors (can contain None)
            max_pattern_length: Maximum pattern length to search (default 50)
            timeout_seconds: Maximum time to spend searching (default 5)
            structural: If True, use structural matching (A→B→A) instead of
                       exact motive matching (motive_1→motive_2→motive_1)

        Returns:
            dict with pattern info: pattern, length, first_occurrence, num_occurrences
        """
        if not sequence or len(sequence) < 2:
            return None

        # CRITICAL: Remove None values - only analyze ACTIVE behaviors
        active_seq = [x for x in sequence if x is not None]

        if len(active_seq) < 4:  # Need at least 4 active behaviors for a pattern
            return None

        # Use structural pattern detection if requested
        if structural:
            return StructuralPatternDetector.find_structural_patterns(
                active_seq, max_pattern_length, timeout_seconds
            )

        # Otherwise use exact motive matching
        return PatternDetector._find_exact_patterns(
            active_seq, max_pattern_length, timeout_seconds
        )

    @staticmethod
    def _find_exact_patterns(active_seq, max_pattern_length, timeout_seconds):
        """Find patterns with exact motive matching."""
        start_time = time.time()

        # Limit sequence length for performance
        if len(active_seq) > 1000:
            active_seq = active_seq[:1000]

        best_pattern = None
        max_length = 0

        max_possible = min(len(active_seq) // 2, max_pattern_length)

        # Try different pattern lengths
        for pattern_len in range(2, max_possible + 1):
            if time.time() - start_time > timeout_seconds:
                break

            # Use a sliding window approach with dictionary
            seen_patterns = {}

            for i in range(len(active_seq) - pattern_len + 1):
                pattern_tuple = tuple(active_seq[i : i + pattern_len])

                if pattern_tuple not in seen_patterns:
                    seen_patterns[pattern_tuple] = [i]
                else:
                    seen_patterns[pattern_tuple].append(i)

            # Find patterns that repeat at least twice
            for pattern_tuple, positions in seen_patterns.items():
                if len(positions) >= 2 and pattern_len > max_length:
                    max_length = pattern_len
                    best_pattern = {
                        "pattern": list(pattern_tuple),
                        "length": pattern_len,
                        "first_occurrence": positions[0],
                        "num_occurrences": len(positions),
                        "occurrence_positions": positions[:10],
                    }

            # Early stopping
            if max_length > 20:
                break

        return best_pattern

    @staticmethod
    def calculate_sequence_similarity(seq1, seq2, structural=False):
        """
        Calculate similarity between two behavior sequences.
        Only compares ACTIVE behaviors (skips None).

        Args:
            seq1, seq2: Lists of behaviors
            structural: If True, compare structural patterns

        Returns:
            dict with similarity metrics
        """
        # Remove None values - only compare active behaviors
        clean_seq1 = [x for x in seq1 if x is not None]
        clean_seq2 = [x for x in seq2 if x is not None]

        if len(clean_seq1) == 0 or len(clean_seq2) == 0:
            return {
                "exact_match_ratio": 0.0,
                "frequency_similarity": 0.0,
                "normalized_edit_similarity": 0.0,
                "structural_similarity": 0.0,
                "overall_similarity": 0.0,
            }

        # 1. Exact match ratio
        min_len = min(len(clean_seq1), len(clean_seq2))
        matches = sum(1 for i in range(min_len) if clean_seq1[i] == clean_seq2[i])
        exact_match_ratio = matches / min_len

        # 2. Frequency distribution similarity (Bhattacharyya coefficient)
        freq1 = Counter(clean_seq1)
        freq2 = Counter(clean_seq2)
        all_behaviors = set(freq1.keys()) | set(freq2.keys())

        total1 = len(clean_seq1)
        total2 = len(clean_seq2)

        bhattacharyya = 0
        for behavior in all_behaviors:
            p1 = freq1.get(behavior, 0) / total1 if total1 > 0 else 0
            p2 = freq2.get(behavior, 0) / total2 if total2 > 0 else 0
            bhattacharyya += np.sqrt(p1 * p2)

        # 3. Edit distance (limited for speed)
        max_compare_len = 200
        seq1_limited = clean_seq1[:max_compare_len]
        seq2_limited = clean_seq2[:max_compare_len]

        edit_distance = PatternDetector._levenshtein_distance(
            seq1_limited, seq2_limited
        )
        max_len = max(len(seq1_limited), len(seq2_limited), 1)
        normalized_edit_distance = 1 - (edit_distance / max_len)

        # 4. Structural similarity (if requested)
        if structural:
            structural_sim = StructuralPatternDetector.calculate_structural_similarity(
                clean_seq1, clean_seq2
            )
        else:
            structural_sim = 0.0

        # Calculate overall
        if structural:
            overall = (
                exact_match_ratio
                + bhattacharyya
                + normalized_edit_distance
                + structural_sim
            ) / 4
        else:
            overall = (exact_match_ratio + bhattacharyya + normalized_edit_distance) / 3

        return {
            "exact_match_ratio": float(exact_match_ratio),
            "frequency_similarity": float(bhattacharyya),
            "normalized_edit_similarity": float(normalized_edit_distance),
            "structural_similarity": float(structural_sim),
            "overall_similarity": float(overall),
        }

    @staticmethod
    def _levenshtein_distance(seq1, seq2):
        """Calculate Levenshtein distance between two sequences."""
        if len(seq1) < len(seq2):
            return PatternDetector._levenshtein_distance(seq2, seq1)

        if len(seq2) == 0:
            return len(seq1)

        previous_row = range(len(seq2) + 1)
        for i, c1 in enumerate(seq1):
            current_row = [i + 1]
            for j, c2 in enumerate(seq2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


# Convenience function
def analyze_pattern_structure(sequence):
    """
    Analyze a behavior sequence and return both exact and structural patterns.

    Args:
        sequence: List of behaviors (can contain None)

    Returns:
        dict with both exact and structural pattern info
    """
    exact_pattern = PatternDetector.find_longest_repeating_pattern(
        sequence, structural=False
    )

    structural_pattern = PatternDetector.find_longest_repeating_pattern(
        sequence, structural=True
    )

    return {"exact_pattern": exact_pattern, "structural_pattern": structural_pattern}

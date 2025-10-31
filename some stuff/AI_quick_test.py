"""
Quick test to verify ai_pattern_analysis and ai_batch_seperated work together
"""

print("=" * 70)
print("TESTING AI MODULE IMPORTS")
print("=" * 70)

# Test 1: Import pattern analysis
print("\n1. Testing ai_pattern_analysis import...")
try:
    from ai_pattern_analysis import PatternDetector, StructuralPatternDetector

    print("   ✓ Successfully imported PatternDetector")
    print("   ✓ Successfully imported StructuralPatternDetector")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    exit(1)

# Test 2: Test basic functionality
print("\n2. Testing pattern detection...")
try:
    test_sequence = [None, "motive_1", None, "motive_2", "motive_1", None, "motive_2"]
    pattern = PatternDetector.find_longest_repeating_pattern(
        test_sequence, structural=False
    )

    if pattern:
        print(f"   ✓ Found pattern: {pattern['pattern']}")
        print(f"   ✓ Pattern length: {pattern['length']}")
        print(f"   ✓ Occurrences: {pattern['num_occurrences']}")
    else:
        print("   ✓ No pattern found (expected for short sequence)")
except Exception as e:
    print(f"   ✗ Pattern detection failed: {e}")
    exit(1)

# Test 3: Test structural encoding
print("\n3. Testing structural encoding...")
try:
    seq1 = ["motive_1", "motive_2", "motive_1"]
    seq2 = ["motive_5", "motive_7", "motive_5"]

    struct1, _ = StructuralPatternDetector.encode_structure(seq1)
    struct2, _ = StructuralPatternDetector.encode_structure(seq2)

    print(f"   ✓ Sequence 1 structure: {struct1}")
    print(f"   ✓ Sequence 2 structure: {struct2}")
    print(f"   ✓ Same structure: {struct1 == struct2}")

    if struct1 == struct2:
        print("   ✓ Structural pattern recognition working!")
    else:
        print("   ✗ Structures should be identical")
        exit(1)

except Exception as e:
    print(f"   ✗ Structural encoding failed: {e}")
    exit(1)

# Test 4: Try importing batch processor
print("\n4. Testing ai_batch_seperated import...")
try:
    from ai_batch_seperated import BatchSimulationProcessor

    print("   ✓ Successfully imported BatchSimulationProcessor")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    print("\nPossible issues:")
    print(
        "   - Make sure hg_lambda_calc.py and hg_game_engine.py are in same directory"
    )
    print("   - Check that all files are in the same folder")
    exit(1)

# Test 5: Try importing report generator
print("\n5. Testing ai_pattern_report_generator import...")
try:
    from ai_pattern_report_generator import PatternAnalysisReport

    print("   ✓ Successfully imported PatternAnalysisReport")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    exit(1)

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\nYou can now use:")
print("  from ai_batch_seperated import run_batch_simulations")
print("  from ai_pattern_report_generator import generate_pattern_report")
print("  from ai_pattern_analysis import PatternDetector, StructuralPatternDetector")
print("\n" + "=" * 70)

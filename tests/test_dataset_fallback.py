"""
Test Dataset Recursive Fallback Behavior.

This test suite validates the peer review findings regarding:
1. Recursive fallback patterns that could cause infinite loops
2. Missing retry counters in dataset classes
3. Silent failure without logging

References:
- Peer Review: PEER_REVIEW.md Section B.1 (lines 155-179)
- Paper: Nature Medicine DOI 10.1038/s41591-025-04133-4
"""
import os
import re
import pytest


class TestRecursiveFallbackPatterns:
    """Test suite for recursive fallback pattern issues in datasets."""

    def test_identify_recursive_fallback_pattern(self):
        """
        PEER REVIEW ISSUE B.1: Recursive fallback on errors.

        Multiple dataset classes use:
            return self.__getitem__((idx + 1) % self.total_len)

        This can cause infinite loops if many consecutive samples are corrupted.

        Reference: PEER_REVIEW.md lines 155-179
        """
        dataset_path = os.path.join(
            os.path.dirname(__file__),
            '..', 'sleepfm', 'models', 'dataset.py'
        )

        with open(dataset_path, 'r') as f:
            content = f.read()

        # Pattern to match recursive __getitem__ calls
        pattern = r'return\s+self\.__getitem__\(\s*\(idx\s*\+\s*1\)\s*%\s*self\.(total_len|len|__len__)'

        matches = re.findall(pattern, content)

        print(f"\nFound {len(matches)} recursive fallback patterns in dataset.py")

        # Document the finding
        assert len(matches) > 0, (
            "Expected to find recursive fallback patterns. "
            "This test documents the peer review finding."
        )

        # Verify this is a potential issue
        print("PEER REVIEW FINDING CONFIRMED: Recursive fallback patterns found")
        print("These could cause infinite loops if all samples in sequence fail.")

    def test_count_affected_dataset_classes(self):
        """Count how many dataset classes have the recursive fallback issue."""
        dataset_path = os.path.join(
            os.path.dirname(__file__),
            '..', 'sleepfm', 'models', 'dataset.py'
        )

        with open(dataset_path, 'r') as f:
            lines = f.readlines()

        # Find class definitions and their __getitem__ methods
        current_class = None
        classes_with_fallback = []
        fallback_pattern = re.compile(r'return\s+self\.__getitem__')

        for i, line in enumerate(lines):
            # Check for class definition
            class_match = re.match(r'^class\s+(\w+)', line)
            if class_match:
                current_class = class_match.group(1)

            # Check for recursive fallback
            if fallback_pattern.search(line) and current_class:
                if current_class not in classes_with_fallback:
                    classes_with_fallback.append(current_class)
                    print(f"Line {i+1}: Found fallback in {current_class}")

        print(f"\nTotal dataset classes with recursive fallback: {len(classes_with_fallback)}")
        print(f"Affected classes: {classes_with_fallback}")

        # Expected to find multiple classes (peer review mentions 4-5)
        assert len(classes_with_fallback) >= 3, (
            f"PEER REVIEW FINDING: Found {len(classes_with_fallback)} classes with "
            f"recursive fallback: {classes_with_fallback}"
        )

    def test_no_retry_counter_exists(self):
        """
        Verify that no retry counter is implemented.

        The peer review recommends adding a maximum retry counter:
            def __getitem__(self, idx, _retry_count=0):
                if _retry_count > 10:
                    raise RuntimeError(...)
        """
        dataset_path = os.path.join(
            os.path.dirname(__file__),
            '..', 'sleepfm', 'models', 'dataset.py'
        )

        with open(dataset_path, 'r') as f:
            content = f.read()

        # Look for retry counter patterns
        has_retry_counter = (
            'retry_count' in content.lower() or
            '_retry' in content.lower() or
            'max_retries' in content.lower()
        )

        # This should be False - no retry counter exists
        assert not has_retry_counter, (
            "PEER REVIEW FINDING CONFIRMED: No retry counter found in dataset code. "
            "Recursive fallback could loop indefinitely."
        )

        print("\nPEER REVIEW FINDING CONFIRMED: No retry counter implemented")
        print("Recommendation: Add _retry_count parameter with max limit")

    def test_missing_error_logging(self):
        """
        Check for proper error logging when data loading fails.

        Peer review notes that most classes fail silently without logging.
        Only SupervisedDiagnosisFullCOXPHWithDemoDataset has logging.
        """
        dataset_path = os.path.join(
            os.path.dirname(__file__),
            '..', 'sleepfm', 'models', 'dataset.py'
        )

        with open(dataset_path, 'r') as f:
            content = f.read()

        # Count logging statements near the recursive fallback
        # A well-implemented fallback should log warnings
        lines = content.split('\n')

        fallback_with_logging = 0
        fallback_without_logging = 0

        for i, line in enumerate(lines):
            if 'return self.__getitem__' in line:
                # Check surrounding lines for logging
                context = '\n'.join(lines[max(0, i-5):i])
                if 'logger' in context or 'logging' in context or 'print' in context:
                    fallback_with_logging += 1
                else:
                    fallback_without_logging += 1

        print(f"\nFallback patterns with logging: {fallback_with_logging}")
        print(f"Fallback patterns without logging: {fallback_without_logging}")

        # Most should be without logging (peer review finding)
        assert fallback_without_logging > fallback_with_logging, (
            "PEER REVIEW FINDING CONFIRMED: Most fallback patterns lack error logging. "
            f"Without logging: {fallback_without_logging}, With logging: {fallback_with_logging}"
        )


class TestDuplicateCollateFunctions:
    """Test for duplicate collate function implementations."""

    def test_count_collate_functions(self):
        """
        PEER REVIEW ISSUE B.4: Duplicate collate functions.

        The peer review notes nearly identical collate_fn implementations:
        - collate_fn
        - sleep_event_finetune_full_collate_fn
        - diagnosis_finetune_full_coxph_collate_fn
        - diagnosis_finetune_full_coxph_with_demo_collate_fn

        Reference: PEER_REVIEW.md lines 239-247
        """
        dataset_path = os.path.join(
            os.path.dirname(__file__),
            '..', 'sleepfm', 'models', 'dataset.py'
        )

        with open(dataset_path, 'r') as f:
            content = f.read()

        # Count collate_fn definitions
        collate_pattern = re.compile(r'^def\s+(\w*collate\w*)\s*\(', re.MULTILINE)
        matches = collate_pattern.findall(content)

        print(f"\nFound {len(matches)} collate functions:")
        for name in matches:
            print(f"  - {name}")

        # Should find multiple collate functions
        assert len(matches) >= 4, (
            f"PEER REVIEW FINDING: Found {len(matches)} collate functions. "
            "These have significant code duplication and should be refactored."
        )

    def test_identify_duplicate_function_names(self):
        """Check for functions defined multiple times with same name."""
        dataset_path = os.path.join(
            os.path.dirname(__file__),
            '..', 'sleepfm', 'models', 'dataset.py'
        )

        with open(dataset_path, 'r') as f:
            content = f.read()

        # Find all function definitions
        func_pattern = re.compile(r'^def\s+(\w+)\s*\(', re.MULTILINE)
        func_names = func_pattern.findall(content)

        # Count occurrences
        from collections import Counter
        func_counts = Counter(func_names)

        duplicates = {name: count for name, count in func_counts.items() if count > 1}

        if duplicates:
            print("\nDuplicate function definitions found:")
            for name, count in duplicates.items():
                print(f"  - {name}: defined {count} times")

            # This is a code quality issue
            print("\nPEER REVIEW FINDING: Duplicate function definitions exist")

        # Document whether duplicates exist
        assert True  # Just document, don't fail


class TestInfiniteLoopSimulation:
    """
    Simulated tests for infinite loop behavior.

    Note: We can't actually test infinite loops without modifying production code,
    but we can analyze the code structure to confirm the risk.
    """

    def test_modulo_wrapping_enables_infinite_loop(self):
        """
        Verify that the modulo pattern enables infinite loops.

        Pattern: return self.__getitem__((idx + 1) % self.total_len)

        If idx starts at 0 and total_len is N, after N iterations
        we return to idx=0, creating an infinite loop.
        """
        dataset_path = os.path.join(
            os.path.dirname(__file__),
            '..', 'sleepfm', 'models', 'dataset.py'
        )

        with open(dataset_path, 'r') as f:
            content = f.read()

        # Check for modulo pattern
        modulo_pattern = r'\(idx\s*\+\s*1\)\s*%\s*self\.\w+'
        has_modulo = re.search(modulo_pattern, content) is not None

        if has_modulo:
            print("\nPEER REVIEW FINDING CONFIRMED:")
            print("Modulo wrapping pattern found: (idx + 1) % self.total_len")
            print("This creates a cycle that enables infinite loops when all samples fail.")
            print("\nRecommended fix:")
            print("  def __getitem__(self, idx, _retry_count=0):")
            print("      if _retry_count > 10:")
            print("          raise RuntimeError(f'Failed after 10 retries from idx {idx}')")
            print("      ...")
            print("      return self.__getitem__((idx + 1) % self.total_len, _retry_count + 1)")

        assert has_modulo, "Expected to find modulo wrapping pattern"

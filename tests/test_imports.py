"""
Test Module Imports.

This test suite validates that all SleepFM modules can be imported correctly.
Also documents reproducibility issues with import paths.
"""
import os
import sys
import pytest


class TestModuleImports:
    """Test that core modules can be imported."""

    def test_models_import(self):
        """Test importing from sleepfm.models.models."""
        from sleepfm.models.models import (
            SetTransformer,
            SleepEventLSTMClassifier,
            Tokenizer,
            AttentionPooling,
        )
        assert SetTransformer is not None
        assert SleepEventLSTMClassifier is not None

    def test_preprocessing_import(self):
        """Test importing preprocessing module."""
        from sleepfm.preprocessing.preprocessing import EDFToHDF5Converter
        assert EDFToHDF5Converter is not None

    def test_utils_import(self):
        """Test importing utils module."""
        from sleepfm.utils import load_config, save_data, load_data
        assert load_config is not None
        assert save_data is not None
        assert load_data is not None

    def test_dataset_import_requires_pythonpath(self):
        """
        Document that dataset module has broken relative imports.

        The dataset.py file uses:
            from utils import load_data, save_data

        This only works if PYTHONPATH includes the sleepfm directory.
        This is a reproducibility issue.
        """
        # This should work because conftest.py adds to sys.path
        try:
            from sleepfm.models.dataset import SetTransformerDataset
            success = True
        except ModuleNotFoundError as e:
            success = False
            print(f"\nImport failed: {e}")
            print("This is a reproducibility issue - relative imports require PYTHONPATH setup")

        assert success, (
            "Dataset import requires PYTHONPATH to include sleepfm directory. "
            "This is a reproducibility issue documented in peer review."
        )


class TestMissingSeedInPretrain:
    """Test for missing random seed setting in pretrain.py."""

    def test_pretrain_missing_seed(self):
        """
        PEER REVIEW ISSUE C.2: Missing seeds in pretrain.py

        While set_seed() is called in fine-tuning scripts, the pretraining
        script (pretrain.py) does NOT set random seeds, affecting reproducibility.

        Reference: PEER_REVIEW.md lines 265-274
        """
        pretrain_path = os.path.join(
            os.path.dirname(__file__),
            '..', 'sleepfm', 'pipeline', 'pretrain.py'
        )

        with open(pretrain_path, 'r') as f:
            content = f.read()

        # Check for seed setting patterns
        has_set_seed = 'set_seed' in content
        has_manual_seed = 'torch.manual_seed' in content
        has_np_seed = 'np.random.seed' in content or 'numpy.random.seed' in content
        has_random_seed = 'random.seed' in content

        any_seed_setting = has_set_seed or has_manual_seed or has_np_seed or has_random_seed

        # This should be False - pretrain.py doesn't set seeds
        assert not any_seed_setting, (
            "PEER REVIEW FINDING CONFIRMED: pretrain.py does NOT set random seeds. "
            "This means pretraining results are NOT reproducible across runs."
        )

        print("\nPEER REVIEW FINDING CONFIRMED:")
        print("pretrain.py does not call set_seed() or any seed-setting function")
        print("Pretraining results are NOT reproducible")

    def test_finetuning_has_seed(self):
        """Verify that finetuning scripts DO set seeds (for comparison)."""
        finetune_path = os.path.join(
            os.path.dirname(__file__),
            '..', 'sleepfm', 'pipeline', 'finetune_diagnosis_coxph.py'
        )

        with open(finetune_path, 'r') as f:
            content = f.read()

        has_set_seed = 'set_seed' in content
        assert has_set_seed, "Fine-tuning scripts should have seed setting"

        print("\nComparison: finetune_diagnosis_coxph.py DOES call set_seed()")


class TestHardcodedValues:
    """Test for hardcoded values that override config."""

    def test_evaluate_sleep_staging_hardcoded(self):
        """
        PEER REVIEW ISSUE B.2: Hardcoded values in evaluation scripts.

        evaluate_sleep_staging.py hardcodes:
        - num_workers = 4  (overrides config)
        - batch_size = 4   (overrides config)

        Reference: PEER_REVIEW.md lines 186-189
        """
        eval_path = os.path.join(
            os.path.dirname(__file__),
            '..', 'sleepfm', 'pipeline', 'evaluate_sleep_staging.py'
        )

        with open(eval_path, 'r') as f:
            content = f.read()

        # Look for hardcoded overrides
        lines = content.split('\n')

        hardcoded_overrides = []
        for i, line in enumerate(lines):
            # Pattern: variable = 4 (simple assignment after config.get())
            if 'num_workers = 4' in line or 'batch_size = 4' in line:
                hardcoded_overrides.append((i + 1, line.strip()))

        if hardcoded_overrides:
            print("\nPEER REVIEW FINDING CONFIRMED: Hardcoded values in evaluate_sleep_staging.py")
            for line_num, code in hardcoded_overrides:
                print(f"  Line {line_num}: {code}")

        assert len(hardcoded_overrides) > 0, (
            "Expected to find hardcoded num_workers or batch_size overrides"
        )

    def test_evaluate_disease_prediction_hardcoded(self):
        """Check for hardcoded values in evaluate_disease_prediction.py."""
        eval_path = os.path.join(
            os.path.dirname(__file__),
            '..', 'sleepfm', 'pipeline', 'evaluate_disease_prediction.py'
        )

        with open(eval_path, 'r') as f:
            content = f.read()

        # Look for batch_size override
        has_batch_override = 'config["batch_size"] = 4' in content or "config['batch_size'] = 4" in content

        if has_batch_override:
            print("\nPEER REVIEW FINDING CONFIRMED: Hardcoded batch_size in evaluate_disease_prediction.py")

        assert has_batch_override, "Expected hardcoded batch_size override"

    def test_hardcoded_paths_in_configs(self):
        """
        PEER REVIEW ISSUE B.2: Hardcoded absolute paths in config files.

        Config files contain Stanford cluster-specific paths like:
        - '/scratch/users/rthapa84/psg_fm/data/data_new_128/'
        - '/oak/stanford/groups/jamesz/rthapa84/psg_fm/'

        Reference: PEER_REVIEW.md lines 191-199
        """
        config_dir = os.path.join(
            os.path.dirname(__file__),
            '..', 'sleepfm', 'configs'
        )

        hardcoded_paths = []

        for filename in os.listdir(config_dir):
            if filename.endswith('.yaml'):
                filepath = os.path.join(config_dir, filename)
                with open(filepath, 'r') as f:
                    content = f.read()

                # Check for Stanford-specific paths
                if '/scratch/' in content or '/oak/' in content:
                    hardcoded_paths.append(filename)

        if hardcoded_paths:
            print("\nPEER REVIEW FINDING CONFIRMED: Hardcoded paths in config files:")
            for f in hardcoded_paths:
                print(f"  - {f}")

        assert len(hardcoded_paths) > 0, (
            "Expected to find hardcoded Stanford cluster paths in configs"
        )

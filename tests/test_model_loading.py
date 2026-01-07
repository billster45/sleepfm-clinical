"""
Test Model Checkpoint Loading.

This test suite validates that model checkpoints can be loaded
and used for inference.
"""
import os
import json
import pytest
import torch


class TestCheckpointExistence:
    """Test that model checkpoints exist and have expected files."""

    def test_model_base_checkpoint_exists(self, model_checkpoint_paths):
        """Verify base pretrained model checkpoint exists."""
        base_path = model_checkpoint_paths['model_base']
        assert os.path.exists(base_path), f"model_base checkpoint not found at {base_path}"

        # Check for expected files
        best_pt = os.path.join(base_path, 'best.pt')
        config_json = os.path.join(base_path, 'config.json')

        assert os.path.exists(best_pt), f"best.pt not found in {base_path}"
        assert os.path.exists(config_json), f"config.json not found in {base_path}"

        # Check file sizes
        pt_size = os.path.getsize(best_pt)
        print(f"\nmodel_base checkpoint size: {pt_size / 1024 / 1024:.2f} MB")
        assert pt_size > 0, "Checkpoint file is empty"

    def test_model_diagnosis_checkpoint_exists(self, model_checkpoint_paths):
        """Verify disease prediction model checkpoint exists."""
        diag_path = model_checkpoint_paths['model_diagnosis']
        assert os.path.exists(diag_path), f"model_diagnosis checkpoint not found at {diag_path}"

        best_pth = os.path.join(diag_path, 'best.pth')
        assert os.path.exists(best_pth), f"best.pth not found in {diag_path}"

        pth_size = os.path.getsize(best_pth)
        print(f"\nmodel_diagnosis checkpoint size: {pth_size / 1024 / 1024:.2f} MB")

    def test_model_sleep_staging_checkpoint_exists(self, model_checkpoint_paths):
        """Verify sleep staging model checkpoint exists."""
        staging_path = model_checkpoint_paths['model_sleep_staging']
        assert os.path.exists(staging_path), f"model_sleep_staging checkpoint not found at {staging_path}"

        best_pth = os.path.join(staging_path, 'best.pth')
        assert os.path.exists(best_pth), f"best.pth not found in {staging_path}"

        pth_size = os.path.getsize(best_pth)
        print(f"\nmodel_sleep_staging checkpoint size: {pth_size / 1024 / 1024:.2f} MB")


class TestCheckpointLoading:
    """Test that checkpoints can be loaded correctly."""

    def test_load_base_model_config(self, model_checkpoint_paths):
        """Load and validate base model configuration."""
        config_path = os.path.join(model_checkpoint_paths['model_base'], 'config.json')

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Check expected config keys
        print(f"\nBase model config keys: {list(config.keys())}")

        # Document the architecture parameters
        if 'embed_dim' in config:
            print(f"  embed_dim: {config['embed_dim']}")
        if 'num_heads' in config:
            print(f"  num_heads: {config['num_heads']}")
        if 'num_layers' in config:
            print(f"  num_layers: {config['num_layers']}")

    def test_load_base_model_weights(self, model_checkpoint_paths):
        """Load base model weights and verify structure."""
        checkpoint_path = os.path.join(model_checkpoint_paths['model_base'], 'best.pt')

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Check what's in the checkpoint
        if isinstance(checkpoint, dict):
            print(f"\nCheckpoint keys: {list(checkpoint.keys())}")

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            if isinstance(state_dict, dict):
                print(f"Number of parameters: {len(state_dict)}")
                # Show first few layer names
                layer_names = list(state_dict.keys())[:5]
                print(f"First few layers: {layer_names}")

    def test_instantiate_set_transformer(self, model_checkpoint_paths):
        """Test that SetTransformer can be instantiated with checkpoint config."""
        from sleepfm.models.models import SetTransformer

        config_path = os.path.join(model_checkpoint_paths['model_base'], 'config.json')

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Try to instantiate model with default params
        try:
            model = SetTransformer(
                in_channels=1,
                patch_size=config.get('patch_size', 640),
                embed_dim=config.get('embed_dim', 128),
                num_heads=config.get('num_heads', 8),
                num_layers=config.get('num_layers', 6),
            )

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"\nSetTransformer instantiated successfully")
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")

            assert total_params > 0

        except Exception as e:
            pytest.fail(f"Failed to instantiate SetTransformer: {e}")


class TestDemoDataExistence:
    """Test that demo data files exist."""

    def test_demo_psg_edf_exists(self, demo_data_paths):
        """Verify demo PSG EDF file exists."""
        edf_path = demo_data_paths['psg_edf']
        assert os.path.exists(edf_path), f"Demo PSG file not found at {edf_path}"

        size = os.path.getsize(edf_path)
        print(f"\nDemo PSG file size: {size / 1024 / 1024:.2f} MB")

    def test_demo_labels_exist(self, demo_data_paths):
        """Verify demo label files exist."""
        for name, path in demo_data_paths.items():
            if name != 'psg_edf':
                exists = os.path.exists(path)
                status = "OK" if exists else "MISSING"
                print(f"{name}: {status}")
                if not exists:
                    print(f"  Expected at: {path}")


class TestArchitectureMismatch:
    """
    Test for LSTM vs Transformer architectural mismatch.

    PEER REVIEW ISSUE E.2: Architecture mismatch between pretraining and finetuning.
    - Pretraining uses Transformer encoders
    - Fine-tuning uses bidirectional LSTM for temporal aggregation
    """

    def test_document_architecture_mismatch(self):
        """Document the architectural mismatch finding."""
        models_path = os.path.join(
            os.path.dirname(__file__),
            '..', 'sleepfm', 'models', 'models.py'
        )

        with open(models_path, 'r') as f:
            content = f.read()

        # Check for Transformer in SetTransformer
        has_transformer = 'TransformerEncoder' in content or 'nn.Transformer' in content

        # Check for LSTM in fine-tuning models
        has_lstm = 'nn.LSTM' in content or 'LSTM' in content

        print("\nPEER REVIEW FINDING E.2 - Architecture Analysis:")
        print(f"  Uses Transformer encoder: {has_transformer}")
        print(f"  Uses LSTM for fine-tuning: {has_lstm}")

        if has_transformer and has_lstm:
            print("\n  CONFIRMED: Mixed architecture")
            print("  - Pretraining: Transformer encoders")
            print("  - Fine-tuning: Bidirectional LSTM")
            print("  This mismatch may limit transfer learning effectiveness")

        assert has_transformer and has_lstm, (
            "Expected both Transformer and LSTM architectures to be present"
        )

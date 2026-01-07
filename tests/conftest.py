"""
Shared pytest fixtures for SleepFM test suite.
"""
import sys
import os
import pytest
import numpy as np
import torch

# Add sleepfm to path to fix relative import issues
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sleepfm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def synthetic_survival_data():
    """Generate synthetic survival data for Cox PH testing."""
    np.random.seed(42)
    n_samples = 100
    n_labels = 5

    # Generate hazard predictions (log hazards)
    hazards = torch.randn(n_samples, n_labels)

    # Generate event times (positive values)
    event_times = torch.abs(torch.randn(n_samples, n_labels)) * 100 + 1

    # Generate event indicators (binary)
    is_event = (torch.rand(n_samples, n_labels) > 0.3).float()

    return hazards, event_times, is_event


@pytest.fixture
def small_batch_data():
    """Generate small batch for testing batch vs global sorting."""
    np.random.seed(42)
    n_samples = 10
    n_labels = 3

    hazards = torch.randn(n_samples, n_labels)
    event_times = torch.abs(torch.randn(n_samples, n_labels)) * 100 + 1
    is_event = (torch.rand(n_samples, n_labels) > 0.3).float()

    return hazards, event_times, is_event


@pytest.fixture
def model_checkpoint_paths():
    """Return paths to model checkpoints."""
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'sleepfm', 'checkpoints')
    return {
        'model_base': os.path.join(base_dir, 'model_base'),
        'model_diagnosis': os.path.join(base_dir, 'model_diagnosis'),
        'model_sleep_staging': os.path.join(base_dir, 'model_sleep_staging'),
    }


@pytest.fixture
def demo_data_paths():
    """Return paths to demo data."""
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'demo_data')
    return {
        'psg_edf': os.path.join(base_dir, 'demo_psg.edf'),
        'sleep_stages': os.path.join(base_dir, 'demo_psg.csv'),
        'demographics': os.path.join(base_dir, 'demo_age_gender.csv'),
        'is_event': os.path.join(base_dir, 'is_event.csv'),
        'time_to_event': os.path.join(base_dir, 'time_to_event.csv'),
    }

# standard imports
import os

# third-party imports
import pytest  # type: ignore
import tensorflow as tf  # type: ignore
from unittest.mock import patch

# module imports
from dataloader import SequenceLoader


@pytest.fixture(name="train_config")
def training_configuration():
    """Configuration parameters for data loader initialization."""
    config = {
        "data_folder": "data",
        "source_suffix": "en",
        "target_suffix": "de",
        "split": "train",
        "tokens_in_batch": 512
    }
    return config


@pytest.fixture(name="dummy_loader")
def dummy_sequence_loader(train_config):
    """Dummy sequence loader with skipped batch generation."""
    with patch.object(SequenceLoader, 'create_batches', return_value=None) as mock_method:
        loader = SequenceLoader(data_folder=train_config["data_folder"], source_suffix=train_config["source_suffix"],
                                target_suffix=train_config["target_suffix"], split=train_config["split"],
                                tokens_in_batch=train_config["tokens_in_batch"])
        mock_method.assert_called_once()
    return loader


@pytest.fixture(name="loader")
def sequence_loader(train_config):
    """Sequence loader in train configuration."""
    loader = SequenceLoader(data_folder=train_config["data_folder"], source_suffix=train_config["source_suffix"],
                            target_suffix=train_config["target_suffix"], split=train_config["split"],
                            tokens_in_batch=train_config["tokens_in_batch"])
    return loader


def test_sequence_loader_initialization(dummy_loader):
    """Test initialization of a dummy data loader."""
    assert isinstance(dummy_loader, SequenceLoader), "mock testing has probably failed"


def test_create_batches(loader):
    """Initialize without mocking `create_batches()`."""
    assert loader.n_batches == len(loader.all_batches)
    assert loader.current_batch == -1


def test_next_dunder(loader):
    """Test iteration over data loader"""
    source_data, target_data, source_lengths, target_lengths = loader.__next__()
    assert isinstance(source_data, tf.Tensor), "source_data is not a tf.Tensor."
    assert isinstance(target_data, tf.Tensor), "target_data is not a tf.Tensor."
    assert isinstance(source_lengths, tf.Tensor), "source_lengths is not a tf.Tensor."
    assert isinstance(target_lengths, tf.Tensor), "target_lengths is not a tf.Tensor."

    assert tf.rank(source_data) == 2, "source_data should be a 2D Tensor."
    assert tf.rank(target_data) == 2, "target_data should be a 2D Tensor."
    assert tf.rank(source_lengths) == 1, "source_lengths should be a 1D Tensor."
    assert tf.rank(target_lengths) == 1, "target_lengths should be a 1D Tensor."

    assert source_data.shape[0] == target_data.shape[0], "Mismatch between number of source and target sequences."
    assert source_lengths.shape[0] == target_lengths.shape[0], "Mismatch between number of source and target lengths."

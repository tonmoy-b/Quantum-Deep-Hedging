import torch
import pytest
from src.utils import get_best_device
from src.architecture import DeepHedgingMLPModel


@pytest.fixture
def default_model():
    device = get_best_device()
    return DeepHedgingMLPModel(device=device)


@pytest.fixture
def custom_model():
    device = get_best_device()
    return DeepHedgingMLPModel(
        input_dim=4, hidden_dims=[128, 64, 32], output_dim=1, device=device
    )


def test_no_hidden_layers():
    """Model should degrade gracefully to a linear model with no hidden layers."""
    device = get_best_device()
    model = DeepHedgingMLPModel(hidden_dims=[], device=device)
    x = torch.randn(8, 4)
    out = model(x)
    assert out.shape == (8, 1)


def test_correct_layers(default_model):
    layers = [layer for layer in default_model.network]
    assert len(layers) == 5
    linear_layers = [m for m in default_model.network if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) == 3
    assert linear_layers[0].in_features == 4
    assert linear_layers[0].out_features == 64
    assert linear_layers[1].in_features == 64
    assert linear_layers[1].out_features == 64
    assert linear_layers[2].in_features == 64
    assert linear_layers[2].out_features == 1

import torch
from pytest_mock import mocker

from src.market_simulations import f01
from src.utils import get_best_device

def test_get_device_with_cuda(mocker):
    # ensure torch.cuda.is_available returns True
    mocker.patch('src.utils.torch.cuda.is_available', return_value=True)
    assert get_best_device() == 'cuda'

def test_get_device_without_cuda(mocker):
    # ensure torch.cuda.is_available returns True
    mocker.patch('src.utils.torch.cuda.is_available', return_value=False)
    assert get_best_device() == 'cpu'


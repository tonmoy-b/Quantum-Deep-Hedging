import torch
import pytest

# from src.market_simulations import pyt_gbm
import src.market_simulations
from src.utils import get_best_device


def test_get_device_with_cuda(mocker):
    # ensure torch.cuda.is_available returns True
    mocker.patch("src.utils.torch.cuda.is_available", return_value=True)
    assert get_best_device() == "cuda"


def test_get_device_without_cuda(mocker):
    # ensure torch.cuda.is_available returns False
    mocker.patch("src.utils.torch.cuda.is_available", return_value=False)
    assert get_best_device() == "cpu"


def test_pyt_gbm_cpu(mocker):
    S0, mu, sigma, T, dt = 100.0, 0.05, 0.2, 1.0, 0.001
    n_paths = 100000
    mocker.patch("src.market_simulations.get_best_device", return_value="cpu")
    assert src.market_simulations.pyt_device_check() == "cpu"


def test_pyt_gbm_cuda(mocker):
    S0, mu, sigma, T, dt = 100.0, 0.05, 0.2, 1.0, 0.001
    n_paths = 100000
    mocker.patch("src.market_simulations.get_best_device", return_value="cuda")
    assert src.market_simulations.pyt_device_check() == "cuda"


def test_pyt_gbm():
    S0, mu, sigma, T, dt = 100.0, 0.05, 0.2, 1.0, 0.001
    n_paths = 100000
    prices = src.market_simulations.pyt_gbm(S0, mu, sigma, T, dt, n_paths)
    assert prices.shape == (n_paths, int(T / dt))  # correct shape
    assert (prices > 0).all()  # prices >= 0
    # final mean should be close to S0 * e^(mu*T) due to the analytical solution to GBM
    expected_mean = S0 * torch.exp(torch.tensor(mu * T))
    assert torch.isclose(prices[:, -1].mean(), expected_mean, rtol=0.01)


@pytest.fixture
def default_heston_model_params():
    return dict(
        n_paths=50000,
        S0=100.0,
        v0=0.04,
        mu=0.05,
        kappa=2.0,
        theta=0.04,
        sigma_v=0.3,
        rho=-0.6,
        T=1.0,
        dt=0.01,
    )


def get_params(default_heston_model_params):
    return default_heston_model_params


def test_heston_simulation_price_positive(default_heston_model_params):
    S, _ = src.market_simulations.heston_simulation(**default_heston_model_params)
    assert (S > 0).all()  # prices must never be negative


def test_heston_simulation_var_nonnegative(default_heston_model_params):
    _, v = src.market_simulations.heston_simulation(**default_heston_model_params)
    assert (v >= 0).all()  # clampin should cause all vars to be >= 0


def test_expected_price(default_heston_model_params):
    # the expected price must be close to S0*exp(mu*T) under risk-neutral conditions
    S, v = src.market_simulations.heston_simulation(**default_heston_model_params)
    params = get_params(default_heston_model_params)
    expected = params["S0"] * torch.exp(torch.tensor(params["mu"] * params["T"]))
    actual = S[:, -1].mean()
    assert torch.isclose(
        actual, expected, rtol=0.02
    ), f"Calculated expected mean is {expected:.2f}, while computed mean is {actual:.2f}"

import torch
from src.utils import get_best_device


def pyt_device_check(device=None):
    if device is None:
        device = get_best_device()
    print(device)
    return device


"""
Creates Geometric Brownian Motion
    - SDE_GBM : dS_t = mu*S_t*dt + sigma*S_t*dW_t
    - discretization of the above SDE : S_{t + delta t} = S_t * [1 + mu*delta t + sigma*sqrt(delta t)*dW]
                                      : or,  S_{t+1} = S_t * (1 + mu*dt + sigma*sqrt(dt)*Z)
S_0: float - starting price 
mu: float - drift of the asset
sigma: float - standard deviation of the asset
T: float - time period
dt: steps per time period
n_paths: int - number of asset-paths to be generated
use_log: bool - generation as prices or log-price 
"""


def pyt_gbm(
    S0: torch.float64,
    mu: torch.float64,
    sigma: torch.float64,
    T: torch.float64,
    dt: torch.float64,
    n_paths: torch.int16,
    use_log: bool = True,
):
    device = get_best_device()
    steps = int(T / dt)
    n_paths = int(n_paths)
    # Generate noise by drawing from std-normal-distribution, hence with mean=0, var=1
    # this is dW Weiner process
    Z = torch.randn(n_paths, steps, device=device)
    # Paths generated in either price and log-price
    if use_log:
        # Log-price: provides numerical stability with non-negative values due to log and also due to additive updates
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * torch.sqrt(torch.tensor(dt)) * Z
        log_increments = drift + diffusion
        # cumulative sum of log-returns using cumsum
        x = torch.log(torch.tensor(S0)) + torch.cumsum(log_increments, dim=1)
        return torch.exp(x)
    else:
        # Price-space: disadvantaged relative to log-prices due to potential for negative values and also multiplicative updates
        # S_{t+1} = S_t * (1 + mu*dt + sigma*sqrt(dt)*Z)
        increments = 1 + mu * dt + sigma * torch.sqrt(torch.tensor(dt)) * Z
        prices = S0 * torch.cumprod(
            increments, dim=1
        )  # return values in proces and not in log-prices
        return prices


def heston_simulation(
    n_paths: int = 256,
    S0: float = 100.0,  # starting stock price
    v0: float = 0.04,  # starting variance (vol^2, so sqrt(0.04)=0.2 or 20% volatility)
    mu: float = 0.05,  # drift of the process
    kappa: float = 2.0,  # mean reversion speed
    theta: float = 0.04,  # long-term variance
    sigma_v: float = 0.3,  # volatility of volatility
    rho: float = -0.7,  # correlation between asset price and variance
    T: float = 1.0,  # overall time horizon
    dt: float = 0.01,  # time step
    scheme: str = "euler",  # 'euler' first then 'milstein' or 'truncated' ##TODO: implement Milstein and Truncated methods
    device="cpu",
):
    """
    Simulates Heston model given asset price paths

    Returns:
        S - torch.Tensor of shape (n_paths, n_steps + 1) containing price paths
        v - torch.Tensor of shape (n_paths, n_steps + 1) containing variance paths
    """
    import torch

    device = get_best_device()
    n_steps = int(T / dt)  # torch.tensor(int (T / dt))
    n_paths = int(n_paths)  # n_paths = torch.tensor(n_paths)
    rho = torch.tensor(rho)

    # generate dW_S, dW_v
    sqrt_dt = torch.sqrt(torch.tensor(dt))
    # independent normal distributions as starting points
    Z1 = torch.randn(n_paths, n_steps, device=device)
    Z2 = torch.randn(n_paths, n_steps, device=device)
    # induce the correlation rho in the Brownian motions
    dW_S = sqrt_dt * Z1  # dW_S = sqrt(dt) * Z1
    dW_v = sqrt_dt * (
        rho * Z1 + torch.sqrt(1 - rho**2) * Z2
    )  # dW_v = sqrt(dt) * (rho * Z1 + sqrt(1 - rho^2) * Z2)
    # initialize tensors to hold prices (S) and variances (v)
    S = torch.zeros(n_paths, n_steps + 1, device=device)
    v = torch.zeros(n_paths, n_steps + 1, device=device)
    S[:, 0] = S0  # set the first price to starting price
    v[:, 0] = v0  # set the first variance to starting variance
    # turn to torch for cuda usage
    kappa = torch.tensor(kappa)
    theta = torch.tensor(theta)
    sigma_v = torch.tensor(sigma_v)
    mu = torch.tensor(mu)

    for t in range(n_steps):
        v_pos = torch.clamp(
            v[:, t], min=0.0
        )  # keep variance positive, since the next column is being updated per loop
        sqrt_v = torch.sqrt(v_pos)
        # update variance as per the Heston equation dv = kappa*(theta - v)*dt + sigma_v*sqrt(v)*dW_v
        v_next = v[:, t] + kappa * (theta - v_pos) * dt + sigma_v * sqrt_v * dW_v[:, t]
        # clamp again before storing to make sure large negative swings in dW dont make it negative
        v[:, t + 1] = torch.clamp(v_next, min=0.0)
        # update stock price as per Heston equation dS = mu*S*dt + sqrt(v)*S*dW_S  <--
        S[:, t + 1] = S[:, t] * (1 + mu * dt + sqrt_v * dW_S[:, t])
    return S, v


if __name__ == "__main__":
    print(get_best_device())
    S0, mu, sigma, T, dt = 100.0, 0.05, 0.2, 1.0, 0.001
    n_paths = 100000
    prices = pyt_gbm(S0, mu, sigma, T, dt, n_paths)

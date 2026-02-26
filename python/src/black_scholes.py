import torch
from torch.distributions import Normal

def get_black_scholes_delta(S: float, 
                        K: float, 
                        T, 
                        t, 
                        r, 
                        sigma,
                        device = 'cpu'):
    '''
    S: current asset price
    K: strike
    T: total time period to expiry
    t: current time step -> note: in years
    r: risk free interest rate
    sigma: volatility, sqrt of variance 
    '''
    device = device
    if T - t <= 0:
        return torch.where(S > K, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
    
    tau = T - t
    # d1 = [ln(S/K) + (r + ((sigma^2)/2)t ] / sigma * sprt(t)
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * torch.sqrt(torch.tensor(tau)))
    # dist ~ N(0,1)
    dist = Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
    return dist.cdf(d1).squeeze()
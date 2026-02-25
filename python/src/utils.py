import torch


def get_best_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class EuropeanCallPayoff:
    def __init__(self, strike: float = 100.0):
        self.strike = strike

    def __call__(self, S_T: torch.Tensor) -> torch.Tensor:
        """
        Args: S_T: Terminal stock prices, shape (n_paths,)
        Returns: Payoffs, shape (n_paths,)
        """
        return torch.relu(S_T - self.strike)  # max(S-K, 0)


def simple_euro_payoff(S_T: torch.Tensor, strike: float, device="cpu") -> torch.tensor:
    return torch.relu(S_T - strike)

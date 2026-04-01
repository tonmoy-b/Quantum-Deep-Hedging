import torch
from torch import nn


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


def save_model_for_inference(
    model: nn.Module,
    parameters: list,
    save_file: str = None,
    overwrite_file: bool = False,
):
    if model is None or parameters is None:
        print("Either model and/or parameters not given. Aborting save...")
        return
    # make sure the model is in eval mode
    try:
        save_file = "../saved_models/saved.pt" if save_file is None else save_file
        model.eval()
        traced_script_module = torch.jit.trace(model, parameters)
        traced_script_module.save(save_file)
        print(f"Model exported successfully to {save_file}")

    except Exception as e:
        print(f"Exception in save_model_for_inference, exception details = {e}")

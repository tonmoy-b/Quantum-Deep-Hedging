from typing import Callable
import torch
from torch import nn
from utils import get_best_device, EuropeanCallPayoff
from market_simulations import heston_simulation


class DeepHedgingMLPModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,  # input to the NN -> [S, v, pnl, old_hedge_ratio ]
        hidden_dims: list[int] = [64, 64],  # dimensions of hidden layers in the MLP
        output_dim: int = 1,  # NN output -> hedging decision
        device: str = get_best_device(),
    ):
        super().__init__()

        # build the NN and hold in class.network var
        layers = []  # list to hold layers of the NN in order
        previous_dim = input_dim  # input is the first layer
        # pack in the hidden layers after input layer
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())  # activation ReLU
            previous_dim = hidden_dim  # following loop iteration will pick up from here
        layers.append(
            nn.Linear(previous_dim, output_dim)
        )  # pack in the output layer of the NN
        self.network = nn.Sequential(*layers).to(
            device=device
        )  # the NN placed in class var network

    def forward(self, x: torch.tensor) -> torch.tensor:
        device = get_best_device()
        x = x.to(device=device)
        return self.network(x)  # simply pass the input forward through the NN


def hedge(
    S_path: torch.tensor,
    v_path: torch.tensor,
    payoff: torch.tensor,
    model: nn.Module,
    transaction_cost: float = 0.001,
) -> tuple[torch.tensor, torch.tensor]:
    """
    This function computes the hedging strategy
    Args
    S of shape [batch_size, n_steps+1] are asset prices gotten by simulation,
        where batch_size is the number of market price paths & variances being considered
        and n_steps is the number of hedging steps per path, with step[0] being initialized to S_0
        the beginning price, which cause the total number of steps to be n_steps+1
    v of shape [batch_size, n_steps+1] are variances gotten from Heston market simulation
    payoffs of shape (batch_size,) are the terminal payoffs per path

    Returns
    pnl of shape (batch_size,) which is the final PnL (for each of the paths)
    hedge_positions over the time_steps
    """
    batch_size, n_steps_plus_1 = S_path.shape
    n_steps = n_steps_plus_1 - 1
    device = get_best_device()

    model.to(device=device)
    S_path.to(device=device)
    v_path.to(device=device)
    payoff.to(device=device)
    # initialize tensors for the processing
    delta = torch.zeros((batch_size, 1), device=device)
    pnl: torch.Tensor = torch.zeros(batch_size, device=device)
    hedge_positions = torch.zeros(batch_size, n_steps, device=device)

    hedging_steps = range(n_steps)
    for step_index in hedging_steps:
        # all prices and variances at step t in the shape (batch_size,)
        S_t = S_path[:, step_index].unsqueeze(1)
        v_t = v_path[:, step_index].unsqueeze(1)
        # network input (input dims of 4) -> [S_t, v_t, delta_{t-1}, pnl_{t-1}]
        # concat at dim=1 for a single vector input to the NN
        net_input: torch.Tensor = torch.cat([S_t, v_t, delta, pnl.unsqueeze(1)], dim=1)
        net_input.to(device=device)
        delta_new = model(net_input)  # get the recommended hedge for this time step
        trade = delta_new - delta
        # cost of trade = transaction cost * | trades * S_t |
        tc_cost = transaction_cost * torch.abs(trade * S_t).squeeze()
        # for loop iterations / steps with a previous step
        if step_index > 0:
            price_change = S_path[:, step_index] - S_path[:, step_index - 1]
            # inner product of
            pnl += delta.squeeze() * price_change
        pnl -= tc_cost  # adjust wealth for accrued transaction costs
        delta = delta_new
        hedge_positions[:, step_index] = delta.squeeze()
    # final pnl  :
    final_price_change = S_path[:, -1] - S_path[:, -2]
    pnl += delta.squeeze() * final_price_change
    pnl -= payoff
    return (pnl, hedge_positions)


def cvar_loss(
    pnl: torch.Tensor, alpha: float = 0.1, device="cpu"
) -> (
    torch.Tensor
):  # scalar return that averages loss over the lowest alpha% of the pnl distribution
    """
    computes Conditional Value At Risk (CVar) as a loss function
    CVar gives losses accrued at worst x% of the tail.

    inputs:
    pnl: tensor holding the current PnL
    alpha: the x% over which to compute avg.

    return average value of alpha% of pnl samples with lowest values
    """
    import torch

    batch_size = pnl[0]  # first value in tensor
    num_rows_worst = max(1, int(alpha * batch_size))
    sorted_pnl_rows, _ = torch.sort(pnl)
    return sorted_pnl_rows[:num_rows_worst].mean()  # return avg of alpha% worst values


def train_deep_hedging_heston(
    model: nn.Module,
    payoff_fn: Callable = EuropeanCallPayoff(),
    n_epochs: int = 100,
    n_paths_per_epoch: int = 1024,
    batch_size: int = 256,
    lr: float = 0.001,
    transaction_cost: float = 0.001,
    alpha: float = 0.05,
    device: str = "cpu",
):
    """Train Deep Hedging model for Heston market simulation"""

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(1):  # n_epochs
        epoch_losses = []
        n_batches = n_paths_per_epoch // batch_size
        batches: iter = range(n_batches)
        for batch_idx in batches:
            # simulate price and variance paths
            S_paths, v_paths = heston_simulation(n_paths=batch_size, device=device)
            payoffs = payoff_fn(S_paths[:, -1])
            print(
                f"For batch_idx-{batch_idx}, payoffs.shape is {payoffs.shape}\n \
            S_paths[:3, :3] is {S_paths[:3, :3] } and v_paths[:3, :3] is { v_paths[:3, :3]}, and \
            payoffs[:3] is {payoffs[:3]}"
            )
            # print(payoffs.shape)
            # execute hedging strategy
            pnl, _ = hedge(
                S_paths,
                v_paths,
                payoffs,
                model=model,
                transaction_cost=transaction_cost,
            )
            # compute CVaR loss
            loss = cvar_loss(pnl, alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = torch.mean(torch.tensor(epoch_losses))
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"At Epoch No. {epoch+1}/{n_epochs}, CVaR Loss is {avg_loss:.4f}")
    return model, losses


if __name__ == "__main__":
    # device = get_best_device()
    # S, v = heston_simulation()
    # model: nn.Module = DeepHedgingMLPModel()
    # payoffs = simple_euro_payoff(S[:, -1], 100.0, "cuda")
    # pnl, hedge_positions = hedge(S, v, payoffs, model)
    # cvar_loss_computed = cvar_loss(pnl, alpha=0.01, device=device)
    model, losses = train_deep_hedging_heston(DeepHedgingMLPModel())

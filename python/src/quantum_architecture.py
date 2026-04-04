import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from src.market_simulations import get_best_device


def get_quantum_device(n_qubits: int = 4):
    """
    This function returns a pennylane quantum device with name=hedging_q_device
    and n_qubits number of qubits
    """
    return qml.device(name="default.qubit", wires=n_qubits)


def heston_simulation_full_truncation(
    n_paths: int = 256,
    S0: float = 100.0,
    v0: float = 0.04,
    mu: float = 0.05,
    kappa: float = 2.0,
    theta: float = 0.04,
    sigma_v: float = 0.3,
    rho: float = -0.7,
    T: float = 1.0,
    dt: float = 0.01,
    device=None,
):
    """
    Refined Heston simulation using Log-Normal price updates
    and Full Truncation for variance stability.
    """
    device = get_best_device() if device is None else device
    n_steps = int(T / dt)

    S = torch.zeros(n_paths, n_steps + 1, device=device)
    v = torch.zeros(n_paths, n_steps + 1, device=device)
    S[:, 0] = S0
    v[:, 0] = v0

    # Brownian Motion Correlation Setup
    # independent normal distributions as starting points
    Z1 = torch.randn(n_paths, n_steps, device=device)
    Z2 = torch.randn(n_paths, n_steps, device=device)
    # induce the correlation rho in the Brownian motions, Cholesky technique
    dW_S = Z1 * torch.sqrt(torch.tensor(dt))
    dW_v = (rho * Z1 + torch.sqrt(1 - torch.tensor(rho) ** 2) * Z2) * torch.sqrt(
        torch.tensor(dt)
    )
    log_S = torch.log(S[:, 0])

    # calculate S_t, v_t per step
    for t in range(n_steps):
        # v_pos = max(v_t, 0), ensure v >= 0
        v_pos = torch.clamp(v[:, t], min=0.0)
        # Euler-Maruyama technique
        dv = kappa * (theta - v_pos) * dt + sigma_v * torch.sqrt(v_pos) * dW_v[:, t]
        v[:, t + 1] = v[:, t] + dv
        # d(ln S) = (mu - 0.5 * v)dt + sqrt(v)dW_S
        d_log_S = (mu - 0.5 * v_pos) * dt + torch.sqrt(v_pos) * dW_S[:, t]
        log_S = log_S + d_log_S
        S[:, t + 1] = torch.exp(log_S)
    return S, v


@qml.qnode(device=get_quantum_device(), interface="torch")
def quantum_orthogonal_hedging_circuit(inputs, network_weights):
    # n_layers: int = 3
    n_qubits: int = 4
    # angle encode the network inputs
    iter_qubits: list[int] = range(4)
    qml.AngleEmbedding(inputs, wires=iter_qubits)
    qml.StronglyEntanglingLayers(weights=network_weights, wires=iter_qubits)
    return [
        qml.expval(qml.PauliZ(i)) for i in range(n_qubits)
    ]  # measure on the Pauli-Z transform of the qubits


class ExpectedShortfallLoss(nn.Module):
    def __init__(self, regularization: float = 0.01, alpha: float = 0.05):
        super(ExpectedShortfallLoss, self).__init__()
        self.alpha: float = alpha
        self.regularization: float = regularization
        self.var_threshold: nn.Parameter = nn.Parameter(torch.tensor(0.0))

    def forward(self, portfolio_returns: torch.Tensor):
        """
        portfolio_returns: torch.Tensor of terminal wealth/returns for each path.
        """
        portfolio_returns = portfolio_returns.double()  # ensure higher precision
        losses = -portfolio_returns  # negated to affect penalization of loss
        clamped_var = self.var_threshold.clamp(
            min=losses.min().item(), max=losses.max().item()
        )
        # tail loss -> max(0, loss - VaR)
        tail_losses = torch.clamp(losses - clamped_var, min=0)
        # Rockafellar-Uryasev formula
        es_loss = self.var_threshold + (1.0 / self.alpha) * torch.mean(tail_losses)
        # L2 regularization on the VaR threshold parameter
        var_penalty = self.regularization * (self.var_threshold ** 2)
        return es_loss + var_penalty 

    def get_var(self):
        return self.var_threshold.item() if self.var_threshold is not None else None


class HybridClassicalQuantumHedger(nn.Module):
    def __init__(self, quantum_layer: qml.qnn.TorchLayer = None, n_layers: int = 3):
        super().__init__()
        self.n_layers = n_layers
        self.n_qubits: int = 4
        # StronglyEntanglingLayers injests in shape: (layers, qubits, 3) -> for layers, qubits, R_x, R_y, R_z.
        self.weights_shape = {"network_weights": (n_layers, 4, 3)}
        self.pre_processing = nn.Linear(
            3, self.n_qubits
        )  # input only S_t, v_t or other params for hedging?
        self.quantum_layer = qml.qnn.TorchLayer(
            quantum_orthogonal_hedging_circuit, self.weights_shape
        )
        self.post_processing = nn.Linear(
            self.n_qubits, 1
        )  # network output - one output for the delta / hedge-ratio

    def forward(self, x):
        x = torch.tanh(self.pre_processing(x))
        x = self.quantum_layer(x)
        return torch.sigmoid(self.post_processing(x))  # Normalized Hedge Ratio


def calculate_terminal_wealth(price_paths, hedge_ratios, transaction_cost=0.0001):
    # price_paths: (batch, n_steps+1), hedge_ratios: (batch, n_steps)
    price_diffs = price_paths[:, 1:] - price_paths[:, :-1]  # (batch, n_steps)
    hedge_gains = hedge_ratios * price_diffs  # (batch, n_steps)
    delta_diffs = torch.abs(
        hedge_ratios[:, 1:] - hedge_ratios[:, :-1]
    )  # (batch, n_steps-1)
    costs = transaction_cost * delta_diffs * price_paths[:, 2:]  # (batch, n_steps-1)
    pnl = hedge_gains[:, 0].unsqueeze(1)  # no transaction cost at t=0
    pnl = torch.cat([pnl, hedge_gains[:, 1:] - costs], dim=1)  # (batch, n_steps)
    total_hedge_pnl = torch.sum(pnl, dim=1)
    strike_price = price_paths[:, 0].mean()
    terminal_payoff = torch.clamp(price_paths[:, -1] - strike_price, min=0)
    return total_hedge_pnl - terminal_payoff


def train_quantum_hedger(
    model: nn.Module, n_epochs: int = 500, batch_size: int = 1024, lr: float = 0.01
):
    device = get_best_device()
    model = model.to(device)
    criterion = ExpectedShortfallLoss(alpha=0.05).to(device)
    # optimizer = optim.Adam(
    #     list(model.parameters()) + list(criterion.parameters()), lr=lr
    # )
    optimizer = optim.Adam([
        {"params": model.pre_processing.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
        {"params": model.quantum_layer.parameters(), "lr": 5e-4}, 
        {"params": model.post_processing.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
        {"params": criterion.parameters(), "lr": 2e-4},
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)
    model.train()
    loss_history = []
    var_history = []
    # create fresh Heston price paths,
    # each epoch will sample from these
    # to control
    master_market_paths_sample_size: int = 25 * 1024
    S_master, v_master = heston_simulation_full_truncation(
        n_paths=master_market_paths_sample_size
    )  # S: (batch, n_steps+1), v: (batch, n_steps+1)
    for epoch in range(n_epochs):
        idx = torch.randint(0, master_market_paths_sample_size, (batch_size,))
        S_batch, v_batch = S_master[idx], v_master[idx]
        S_input = S_batch[:, :-1].reshape(-1, 1) / 100.0  # Normalized Price
        v_input = v_batch[:, :-1].reshape(-1, 1) / 0.04  # Normalized Vol
        n_steps = S_batch.shape[1] - 1
        time_steps = torch.linspace(1.0, 0.0, n_steps, device=S_batch.device) 
        time_input = time_steps.unsqueeze(0).expand(batch_size, -1).reshape(-1, 1) # shape : (batch_size*n_steps, 1)
        # normalize with atan for better fidelity with angle enc.
        # S_normalized = 2 * torch.atan(S_batch[:, :-1] / 100.0)
        # v_normalized = 2 * torch.atan(v_batch[:, :-1] / 0.04)
        inputs = torch.cat([S_input, v_input, time_input], dim=-1).float()  # shape : (batch_size*n_steps, 3)
        #inputs = torch.stack([S_normalized, v_normalized], dim=-1).view(-1, 2).float()
        optimizer.zero_grad()
        deltas = model(inputs).view(batch_size, -1)  # re-shape to (batch, n_steps)
        wealth = calculate_terminal_wealth(S_batch, deltas)
        loss = criterion(wealth)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss.item())
        # record
        current_loss = loss.item()
        current_var = criterion.get_var()
        loss_history.append(current_loss)
        var_history.append(current_var)
        if epoch % 5 == 0:
            print(
                f"Epoch {epoch:03d} | ES Loss: {current_loss:.4f} | VaR Est: {current_var:.6f}"
            )
    return model, loss_history, var_history, S_master, v_master


def plot_training_results(loss_history, var_history):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # ES block
    color = "tab:blue"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Expected Shortfall (Loss)", color=color)
    ax1.plot(loss_history, color=color, label="ES Loss", linewidth=2)
    ax1.tick_params(axis="y", labelcolor=color)
    # Var block
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Estimated VaR", color=color)
    ax2.plot(var_history, color=color, linestyle="--", label="VaR Threshold")
    ax2.tick_params(axis="y", labelcolor=color)
    # grid out
    plt.title("Hybrid Quantum Hedging Training Progress")
    fig.tight_layout()
    plt.grid(alpha=0.3)
    plt.show()

def plot_wealth(model, S_master, v_master, batch_size=256, epoch_interval=50, n_epochs=500):
    """
    Plots the distribution of terminal wealth across paths, sampled every epoch_interval epochs.
    """
    model.eval()
    device = next(model.parameters()).device
    n_snapshots = n_epochs // epoch_interval
    fig, axes = plt.subplots(
        n_snapshots, 1, figsize=(12, 3 * n_snapshots), sharex=True
    )
    if n_snapshots == 1:
        axes = [axes]
    with torch.no_grad():
        for i, epoch in enumerate(range(0, n_epochs, epoch_interval)):
            idx = torch.randint(0, S_master.shape[0], (batch_size,))
            S_batch = S_master[idx].to(device)
            v_batch = v_master[idx].to(device)
            n_steps = S_batch.shape[1] - 1
            time_steps = torch.linspace(1.0, 0.0, n_steps, device=device)
            time_input = time_steps.unsqueeze(0).expand(batch_size, -1).reshape(-1, 1)
            S_input = S_batch[:, :-1].reshape(-1, 1) / 100.0
            v_input = v_batch[:, :-1].reshape(-1, 1) / 0.04
            inputs = torch.cat([S_input, v_input, time_input], dim=-1).float()
            deltas = model(inputs).view(batch_size, -1)
            wealth = calculate_terminal_wealth(S_batch, deltas).cpu().numpy()
            ax = axes[i]
            ax.hist(wealth, bins=40, color="steelblue", alpha=0.7, edgecolor="white")
            ax.axvline(wealth.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {wealth.mean():.2f}")
            ax.axvline(0, color="black", linestyle=":", linewidth=1.0, label="Break-even")
            ax.set_ylabel("Count")
            ax.set_title(f"Epoch {epoch} — mean: {wealth.mean():.2f}, std: {wealth.std():.2f}")
            ax.legend(fontsize=8)
    axes[-1].set_xlabel("Terminal Wealth")
    fig.suptitle("Terminal Wealth Distribution per Epoch Snapshot", fontsize=14, y=1.01)
    fig.tight_layout()
    plt.show()


def full_train_loop():
    q_model = HybridClassicalQuantumHedger(n_layers=3)
    trained_model, losses, vars_est, S_master, v_master = train_quantum_hedger(q_model)
    plot_training_results(losses, vars_est)
    plot_wealth(trained_model, S_master, v_master)
    return trained_model


if __name__ == "__main__":
    print("twiity!!")

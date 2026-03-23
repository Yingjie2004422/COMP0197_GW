# test.py
# Evaluation script for the trained probabilistic ECG forecaster.
#
# Run:  python test.py
#
# What this script does
# ---------------------
# 1. Loads the best checkpoint from MODEL_DIR.
# 2. Evaluates on the validation set, reporting MSE and NLL.
# 3. Plots forecast predictions with uncertainty bands (±1σ, ±2σ).
# 4. Decomposes uncertainty into aleatoric (data noise) and epistemic
#    (model uncertainty via MC Dropout) components.
# 5. Saves all figures to RESULTS_DIR.
#
# GenAI assistance: used to draft the MC Dropout loop and plot layout;
# the uncertainty decomposition formula was verified against Kendall &
# Gal (2017) "What Uncertainties Do We Need in Bayesian Deep Learning?"

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DATA_FOLDER, INPUT_LEN, FORECAST_LEN, STRIDE,
    BATCH_SIZE, TRAIN_VAL_SPLIT, SEED,
    MODEL_DIR, RESULTS_DIR, MC_SAMPLES,
)
from dataset import get_dataloaders
from model import ProbabilisticLSTM, gaussian_nll_loss


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> ProbabilisticLSTM:
    """Reconstruct the model from a checkpoint saved by train.py."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt["config"]
    model = ProbabilisticLSTM(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        forecast_len=cfg["forecast_len"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(
        f"[test] Loaded checkpoint from epoch {ckpt['epoch']} "
        f"(val NLL: {ckpt['val_loss']:.4f})"
    )
    return model


# ---------------------------------------------------------------------------
# MC Dropout inference
# ---------------------------------------------------------------------------

def mc_dropout_predict(
    model: ProbabilisticLSTM,
    x: torch.Tensor,
    n_samples: int = MC_SAMPLES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run *n_samples* stochastic forward passes with dropout enabled.

    Returns
    -------
    combined_mean : (batch, forecast_len)  — mean of predicted means
    total_std     : (batch, forecast_len)  — sqrt(epistemic + aleatoric var)
    epistemic_var : (batch, forecast_len)  — variance of predicted means
    aleatoric_var : (batch, forecast_len)  — mean of predicted variances
    """
    # model.train() keeps dropout active; gradients are not needed
    model.train()
    all_means = []
    all_vars  = []

    with torch.no_grad():
        for _ in range(n_samples):
            out   = model(x)                         # (batch, forecast_len, 2)
            mu    = out[..., 0]                      # (batch, forecast_len)
            sigma2 = torch.exp(out[..., 1])          # aleatoric variance
            all_means.append(mu)
            all_vars.append(sigma2)

    all_means = torch.stack(all_means)   # (n_samples, batch, forecast_len)
    all_vars  = torch.stack(all_vars)    # (n_samples, batch, forecast_len)

    # Law of total variance:
    #   E[Var[y|θ]]  = mean aleatoric variance across samples
    #   Var[E[y|θ]]  = variance of means across samples (epistemic)
    aleatoric_var = all_vars.mean(dim=0)             # (batch, forecast_len)
    epistemic_var = all_means.var(dim=0)             # (batch, forecast_len)
    combined_mean = all_means.mean(dim=0)            # (batch, forecast_len)
    total_std     = torch.sqrt(aleatoric_var + epistemic_var)

    return combined_mean, total_std, epistemic_var, aleatoric_var


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: ProbabilisticLSTM,
    loader,
    device: torch.device,
) -> tuple[float, float]:
    """Return (mean_NLL, mean_MSE) over the full loader using a single
    deterministic forward pass (dropout disabled)."""
    model.eval()
    total_nll = total_mse = 0.0
    n = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            total_nll += gaussian_nll_loss(output, y).item()
            total_mse += ((output[..., 0] - y) ** 2).mean().item()
            n += 1

    return total_nll / n, total_mse / n


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_predictions(
    model: ProbabilisticLSTM,
    loader,
    device: torch.device,
    n_examples: int = 6,
) -> None:
    """Plot *n_examples* forecast examples with uncertainty bands and save
    to RESULTS_DIR/predictions.png."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Draw one batch — use val loader which is not shuffled
    x_batch, y_batch = next(iter(loader))
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    mean, total_std, epi_var, ale_var = mc_dropout_predict(model, x_batch)

    mean      = mean.cpu().numpy()
    total_std = total_std.cpu().numpy()
    epi_std   = np.sqrt(epi_var.cpu().numpy())
    ale_std   = np.sqrt(ale_var.cpu().numpy())
    x_np      = x_batch.cpu().squeeze(-1).numpy()
    y_np      = y_batch.cpu().numpy()

    n = min(n_examples, len(x_np))
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]

    input_t    = np.arange(INPUT_LEN)
    forecast_t = np.arange(INPUT_LEN, INPUT_LEN + FORECAST_LEN)

    for i, ax in enumerate(axes):
        ax.plot(input_t,    x_np[i],   color="steelblue", lw=0.8, alpha=0.9,  label="Input ECG")
        ax.plot(forecast_t, y_np[i],   color="green",     lw=1.2,              label="Ground Truth")
        ax.plot(forecast_t, mean[i],   color="crimson",   lw=1.2, ls="--",     label="Predicted μ")

        ax.fill_between(
            forecast_t,
            mean[i] - total_std[i],
            mean[i] + total_std[i],
            alpha=0.30, color="crimson", label="±1σ total",
        )
        ax.fill_between(
            forecast_t,
            mean[i] - 2 * total_std[i],
            mean[i] + 2 * total_std[i],
            alpha=0.12, color="crimson", label="±2σ total",
        )

        ax.axvline(INPUT_LEN, color="gray", ls=":", lw=1)
        ax.set_title(
            f"Sample {i + 1}  |  Aleatoric σ: {ale_std[i].mean():.4f}"
            f"  |  Epistemic σ: {epi_std[i].mean():.4f}"
        )
        ax.set_xlabel("Timestep (samples)")
        ax.set_ylabel("Normalised ECG")
        ax.legend(loc="upper right", fontsize=8, ncol=3)

    plt.suptitle(
        "Probabilistic ECG Forecasting — MC Dropout Uncertainty", fontsize=13, y=1.005
    )
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "predictions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[test] Predictions plot saved → {out}")


def plot_uncertainty_horizon(
    model: ProbabilisticLSTM,
    loader,
    device: torch.device,
    max_batches: int = 40,
) -> None:
    """Show how aleatoric and epistemic uncertainty grow across the forecast
    horizon and save to RESULTS_DIR/uncertainty_horizon.png."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_ale, all_epi = [], []

    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        _, _, epi_var, ale_var = mc_dropout_predict(model, x, n_samples=20)
        all_ale.append(ale_var.cpu().numpy())
        all_epi.append(epi_var.cpu().numpy())

    ale = np.sqrt(np.concatenate(all_ale, axis=0).mean(axis=0))  # (forecast_len,)
    epi = np.sqrt(np.concatenate(all_epi, axis=0).mean(axis=0))

    t = np.arange(FORECAST_LEN) / 360  # convert samples → seconds

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, ale, color="orange", label="Aleatoric σ (data noise)")
    ax.plot(t, epi, color="purple", label="Epistemic σ (model uncertainty)")
    ax.fill_between(t, 0, ale, alpha=0.15, color="orange")
    ax.fill_between(t, 0, epi, alpha=0.15, color="purple")
    ax.set_xlabel("Forecast horizon (seconds)")
    ax.set_ylabel("Standard Deviation (normalised units)")
    ax.set_title("Uncertainty Decomposition Across Forecast Horizon")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "uncertainty_horizon.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[test] Uncertainty horizon plot saved → {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path  = os.path.join(MODEL_DIR, "best_model.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint found at '{ckpt_path}'. "
            "Please run  python train.py  first."
        )

    model = load_model(ckpt_path, device)

    _, val_loader = get_dataloaders(
        data_folder=DATA_FOLDER,
        input_len=INPUT_LEN,
        forecast_len=FORECAST_LEN,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
        split=TRAIN_VAL_SPLIT,
        seed=SEED,
    )

    # --- Quantitative evaluation ---
    nll, mse = evaluate(model, val_loader, device)
    print(f"\n[test] ── Evaluation Results ──────────────────")
    print(f"[test]   MSE (deterministic μ vs truth) : {mse:.6f}")
    print(f"[test]   NLL (Gaussian, deterministic)  : {nll:.6f}")
    print(f"[test] ───────────────────────────────────────\n")

    # --- Visual results ---
    plot_predictions(model, val_loader, device, n_examples=6)
    plot_uncertainty_horizon(model, val_loader, device)

    print(f"\n[test] All results saved to '{RESULTS_DIR}/'")


if __name__ == "__main__":
    main()

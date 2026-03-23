# train.py
# Full training pipeline for the probabilistic ECG forecaster.
#
# Run:  python train.py
#
# What this script does
# ---------------------
# 1. Loads all MIT-BIH records from DATA_FOLDER and creates sliding-window
#    train / validation DataLoaders (record-level split, no temporal leakage).
# 2. Instantiates ProbabilisticLSTM and trains it with the Gaussian NLL loss.
# 3. Applies gradient clipping and a ReduceLROnPlateau learning-rate schedule.
# 4. Saves the best checkpoint (lowest validation NLL) to MODEL_DIR.
# 5. Writes a loss-curve plot to RESULTS_DIR.
#
# GenAI assistance: used to draft the training loop boilerplate; gradient
# clipping, LR scheduling, and checkpoint logic were reviewed and adjusted
# by the team for correctness.

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on headless machines
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (
    DATA_FOLDER, INPUT_LEN, FORECAST_LEN, STRIDE,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
    TRAIN_VAL_SPLIT, SEED, MODEL_DIR, RESULTS_DIR,
)
from dataset import get_dataloaders
from model import ProbabilisticLSTM, gaussian_nll_loss


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, training: bool) -> float:
    """Run one full pass over *loader*.  If *training* is True, also
    back-propagates and updates weights.  Returns mean NLL for the epoch."""
    model.train() if training else model.eval()
    total_loss = 0.0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if training:
                optimizer.zero_grad()

            output = model(x)                        # (batch, forecast_len, 2)
            loss   = gaussian_nll_loss(output, y)

            if training:
                loss.backward()
                # Gradient clipping prevents exploding gradients in the LSTM
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train() -> None:
    # Reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # --- Data ---
    train_loader, val_loader = get_dataloaders(
        data_folder=DATA_FOLDER,
        input_len=INPUT_LEN,
        forecast_len=FORECAST_LEN,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
        split=TRAIN_VAL_SPLIT,
        seed=SEED,
    )

    # --- Model ---
    model = ProbabilisticLSTM(
        input_size=1,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        forecast_len=FORECAST_LEN,
        dropout=DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model parameters: {n_params:,}")

    # --- Optimiser and scheduler ---
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    # --- Checkpoint / results directories ---
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ckpt_path = os.path.join(MODEL_DIR, "best_model.pt")

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    print(f"\n[train] Starting training for {NUM_EPOCHS} epochs\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, training=True)
        val_loss   = run_epoch(model, val_loader,   optimizer, device, training=False)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        flag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    # Store architecture config so test.py can rebuild the model
                    "config": {
                        "input_size":  1,
                        "hidden_size": HIDDEN_SIZE,
                        "num_layers":  NUM_LAYERS,
                        "forecast_len": FORECAST_LEN,
                        "dropout":     DROPOUT,
                    },
                },
                ckpt_path,
            )
            flag = "  ← best"

        print(
            f"Epoch [{epoch:3d}/{NUM_EPOCHS}] "
            f"Train NLL: {train_loss:.4f}  "
            f"Val NLL: {val_loss:.4f}{flag}"
        )

    # --- Loss curve ---
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, label="Train NLL")
    ax.plot(val_losses,   label="Val NLL")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative Log-Likelihood")
    ax.set_title("Training Progress")
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "loss_curves.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"\n[train] Done.  Best val NLL: {best_val_loss:.4f}")
    print(f"[train] Checkpoint saved to : {ckpt_path}")
    print(f"[train] Loss curve saved to : {fig_path}")


if __name__ == "__main__":
    train()

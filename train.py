# train.py
# Full training pipeline for the annotation-aware probabilistic ECG forecaster.
#
# Run:  python train.py
#
# What this script does
# ---------------------
# 1. Loads all MIT-BIH records (signal + beat annotations) and creates
#    sliding-window train / validation DataLoaders (record-level split).
# 2. Instantiates ProbabilisticLSTM and trains it with a combined loss:
#      L = NLL_signal  +  RISK_LAMBDA * BCE_arrhythmia_risk
# 3. Applies gradient clipping and ReduceLROnPlateau scheduling.
# 4. Saves the best checkpoint (lowest total validation loss) to MODEL_DIR.
# 5. Writes a dual loss-curve plot to RESULTS_DIR.
#
# GenAI assistance: used to draft the training-loop boilerplate; the dual-loss
# formulation, gradient clipping, and checkpoint logic were reviewed and
# adjusted by the team.

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
    EMBED_DIM, NUM_BEAT_CLASSES, RISK_LAMBDA,
    TRAIN_VAL_SPLIT, SEED, MODEL_DIR, RESULTS_DIR,
)
from dataset import get_dataloaders
from model import ProbabilisticLSTM, combined_loss


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def run_epoch(
    model:       ProbabilisticLSTM,
    loader,
    optimizer,
    device:      torch.device,
    pw_tensor:   torch.Tensor,      # pos_weight for BCE on this device
    training:    bool,
) -> tuple[float, float, float]:
    """One full pass over *loader*.

    Returns (mean_total_loss, mean_nll, mean_risk_bce) for the epoch.
    """
    model.train() if training else model.eval()
    total_loss = total_nll = total_risk = 0.0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x_sig, x_ann, y_sig, y_risk in loader:
            x_sig  = x_sig.to(device)
            x_ann  = x_ann.to(device)
            y_sig  = y_sig.to(device)
            y_risk = y_risk.to(device)

            if training:
                optimizer.zero_grad()

            sig_out, risk_logit = model(x_sig, x_ann)

            loss, nll_val, risk_val = combined_loss(
                sig_out, risk_logit, y_sig, y_risk, pw_tensor, RISK_LAMBDA
            )

            if training:
                loss.backward()
                # Gradient clipping prevents exploding gradients in the LSTM
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            total_nll  += nll_val
            total_risk += risk_val

    n = len(loader)
    return total_loss / n, total_nll / n, total_risk / n


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # --- Data ---
    train_loader, val_loader, pos_weight = get_dataloaders(
        data_folder=DATA_FOLDER,
        input_len=INPUT_LEN,
        forecast_len=FORECAST_LEN,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
        split=TRAIN_VAL_SPLIT,
        seed=SEED,
    )
    # pos_weight as a device tensor for BCEWithLogitsLoss
    pw_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    # --- Model ---
    model = ProbabilisticLSTM(
        input_size      = 1,
        hidden_size     = HIDDEN_SIZE,
        num_layers      = NUM_LAYERS,
        forecast_len    = FORECAST_LEN,
        dropout         = DROPOUT,
        embed_dim       = EMBED_DIM,
        num_beat_classes= NUM_BEAT_CLASSES,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model parameters: {n_params:,}")

    # --- Optimiser and scheduler ---
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    # --- Output directories ---
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ckpt_path = os.path.join(MODEL_DIR, "best_model.pt")

    best_val_loss = float("inf")
    history = {"train_total": [], "val_total": [],
               "train_nll": [],   "val_nll": [],
               "train_risk": [],  "val_risk": []}

    print(f"\n[train] Starting training for {NUM_EPOCHS} epochs\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_total, tr_nll, tr_risk = run_epoch(
            model, train_loader, optimizer, device, pw_tensor, training=True)
        va_total, va_nll, va_risk = run_epoch(
            model, val_loader,   optimizer, device, pw_tensor, training=False)

        for k, v in zip(
            ["train_total","val_total","train_nll","val_nll","train_risk","val_risk"],
            [tr_total, va_total, tr_nll, va_nll, tr_risk, va_risk],
        ):
            history[k].append(v)

        scheduler.step(va_total)

        flag = ""
        if va_total < best_val_loss:
            best_val_loss = va_total
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "pos_weight": pos_weight,
                    # Full architecture config so test.py can rebuild the model
                    "config": {
                        "input_size":       1,
                        "hidden_size":      HIDDEN_SIZE,
                        "num_layers":       NUM_LAYERS,
                        "forecast_len":     FORECAST_LEN,
                        "dropout":          DROPOUT,
                        "embed_dim":        EMBED_DIM,
                        "num_beat_classes": NUM_BEAT_CLASSES,
                    },
                },
                ckpt_path,
            )
            flag = "  << best"

        print(
            f"Epoch [{epoch:3d}/{NUM_EPOCHS}]  "
            f"Total: {tr_total:.4f}/{va_total:.4f}  "
            f"NLL: {tr_nll:.4f}/{va_nll:.4f}  "
            f"Risk BCE: {tr_risk:.4f}/{va_risk:.4f}"
            f"{flag}"
        )

    # --- Dual loss curves ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, NUM_EPOCHS + 1)

    for ax, train_key, val_key, title in zip(
        axes,
        ["train_total", "train_nll",  "train_risk"],
        ["val_total",   "val_nll",    "val_risk"],
        ["Total Loss",  "Signal NLL", "Risk BCE"],
    ):
        ax.plot(epochs, history[train_key], label="Train")
        ax.plot(epochs, history[val_key],   label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.suptitle("Training Progress", fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "loss_curves.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"\n[train] Done.  Best val total loss: {best_val_loss:.4f}")
    print(f"[train] Checkpoint : {ckpt_path}")
    print(f"[train] Loss curves: {fig_path}")


if __name__ == "__main__":
    train()

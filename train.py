# train.py
# Full training pipeline for the annotation-aware probabilistic ECG forecaster.
#
# Run:  python train.py
#
# Improvements over baseline
# --------------------------
#   Data augmentation            : random noise / amplitude scale / baseline wander applied
#                                  per-batch during training only (not validation).
#   CosineAnnealingWarmRestarts  : LR restarts for better exploration.
#   AdamW + weight decay         : decoupled L2 regularisation.
#   Early stopping               : halts when val loss stagnates; saves best checkpoint.
#   Beta-NLL / CRPS loss         : passed through combined_loss; prevents variance collapse.
#   Bidirectional LSTM           : BiLSTM encoder with projection layer.
#   Multi-head temporal attention: toggled via USE_ATTENTION in config.py.
#   Seq2Seq decoder              : autoregressive decoder with teacher forcing.
#   Dual-lead input              : INPUT_CHANNELS=2 ECG leads.
#   HRV features                 : SDNN, RMSSD, pNN50 fused into hidden state.
#   MDN (K>1)                    : Mixture Density Network signal head.
#   Deep Ensemble                : N_ENSEMBLE models trained with different seeds.
#   Curriculum learning          : first CURRICULUM_EPOCHS epochs use normal-only windows.
#   Focal loss                   : risk head trained with focal BCE when USE_FOCAL_LOSS=True.
#   Oversampling                 : WeightedRandomSampler for arrhythmia windows.
#   Label smoothing              : soft targets for risk BCE loss.
#   Temperature scaling          : calibrate risk head on val set after training.
#
# Ablation experiments
# --------------------
# Edit flags in config.py and re-run.  Models are saved as model_0.pt,
# model_1.pt, ... in MODEL_DIR so ensemble members are preserved.
#
# GenAI assistance: used to draft training-loop boilerplate; ensemble,
# curriculum, focal loss, and MDN wiring were reviewed and adjusted by the team.
# Extended with bidirectional LSTM, multi-head attention, seq2seq decoder,
# dual-lead input, HRV fusion, warm restarts, label smoothing, temperature scaling.

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Subset

from config import (
    DATA_FOLDER, INPUT_LEN, FORECAST_LEN, STRIDE,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    HIDDEN_SIZE, NUM_LAYERS, DROPOUT, USE_LAYER_NORM,
    EMBED_DIM, NUM_BEAT_CLASSES, RISK_LAMBDA, BETA_NLL,
    USE_ATTENTION, USE_RR_FEATURES, USE_RISK_HEAD, DETERMINISTIC,
    AUGMENT_TRAIN, TRAIN_VAL_SPLIT, SEED, MODEL_DIR, RESULTS_DIR,
    EARLY_STOPPING_PATIENCE, USE_MDN, K_MDN, N_ENSEMBLE,
    USE_FOCAL_LOSS, USE_CRPS_LOSS, CURRICULUM_EPOCHS,
    LR_T0, LR_T_MULT, INPUT_CHANNELS,
)
from dataset import get_dataloaders
from model import ProbabilisticLSTM, combined_loss


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

def augment(x_sig: torch.Tensor) -> torch.Tensor:
    """Apply randomised augmentations to a training batch of ECG windows.

    Applied per-batch during training only — validation data is never augmented.

    Augmentations (each applied independently with stated probability):
        Gaussian noise   (p=0.5) : σ_noise ∈ U[0, 0.05]
        Amplitude scale  (p=0.3) : scale   ∈ U[0.90, 1.10]
        Baseline wander  (p=0.3) : low-frequency sine, amp ∈ U[0, 0.10]
    """
    if torch.rand(1).item() < 0.5:
        noise = torch.rand(1).item() * 0.05
        x_sig = x_sig + torch.randn_like(x_sig) * noise

    if torch.rand(1).item() < 0.3:
        scale = 0.90 + torch.rand(1).item() * 0.20
        x_sig = x_sig * scale

    if torch.rand(1).item() < 0.3:
        seq_len = x_sig.shape[1]
        t       = torch.linspace(0, 4 * np.pi, seq_len, device=x_sig.device)
        freq    = 0.10 + torch.rand(1).item() * 0.40
        amp     = torch.rand(1).item() * 0.10
        wander  = amp * torch.sin(freq * t).view(1, -1, 1)
        x_sig   = x_sig + wander

    return x_sig


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when val loss has not improved for *patience* epochs.

    Saves the best checkpoint on every improvement so the weights on disk
    always correspond to the lowest validation loss seen.
    """

    def __init__(self, patience: int, ckpt_path: str) -> None:
        self.patience  = patience
        self.ckpt_path = ckpt_path
        self.counter   = 0
        self.best_loss = float("inf")

    @property
    def improved(self) -> bool:
        return self.counter == 0

    def step(
        self,
        val_loss:  float,
        model:     ProbabilisticLSTM,
        optimizer,
        epoch:     int,
        extra:     dict,
    ) -> bool:
        """Returns True when training should stop."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":             self.best_loss,
                    **extra,
                },
                self.ckpt_path,
            )
        else:
            self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def fit_temperature(model, val_loader, device) -> float:
    """Fit a scalar temperature on val set to calibrate risk head. Returns T."""
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x_sig, x_ann, x_feat, y_sig, y_risk, _, x_hrv in val_loader:
            x_sig  = x_sig.to(device);  x_ann  = x_ann.to(device)
            x_feat = x_feat.to(device); x_hrv  = x_hrv.to(device)
            _, risk_logit = model(x_sig, x_ann, x_feat, x_hrv=x_hrv)
            if risk_logit is None:
                return 1.0
            all_logits.append(risk_logit.squeeze(-1).cpu())
            all_labels.append(y_risk.cpu())

    if not all_logits:
        return 1.0

    logits_t = torch.cat(all_logits).float()
    labels_t = torch.cat(all_labels).float()

    temperature = nn.Parameter(torch.ones(1))
    optimizer   = torch.optim.LBFGS([temperature], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits_t / temperature.clamp(min=0.01), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = temperature.item()
    print(f"  Temperature scaling: T={T:.4f}")
    return T


# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------

def run_epoch(
    model:     ProbabilisticLSTM,
    loader:    DataLoader,
    optimizer,
    device:    torch.device,
    pw_tensor: torch.Tensor,
    training:  bool,
    K:         int,
) -> tuple[float, float, float]:
    """One full pass over *loader*.  Returns (total, signal, risk) mean losses.

    The batch is a 7-tuple; y_beat_type (6th element) is unpacked but ignored
    during training — it is used only at test time for per-beat-type metrics.
    x_hrv (7th element) is the HRV feature vector passed to the model.
    """
    model.train() if training else model.eval()
    total = sig_sum = risk_sum = 0.0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x_sig, x_ann, x_feat, y_sig, y_risk, _, x_hrv in loader:
            x_sig  = x_sig.to(device)
            x_ann  = x_ann.to(device)
            x_feat = x_feat.to(device)
            y_sig  = y_sig.to(device)
            y_risk = y_risk.to(device)
            x_hrv  = x_hrv.to(device)

            # Augment signal during training only
            if training and AUGMENT_TRAIN:
                x_sig = augment(x_sig)

            if training:
                optimizer.zero_grad()

            sig_out, risk_logit = model(x_sig, x_ann, x_feat, x_hrv=x_hrv, y_signal=y_sig)

            loss, sig_val, risk_val = combined_loss(
                sig_out, risk_logit, y_sig, y_risk,
                pw_tensor, RISK_LAMBDA, BETA_NLL,
                K=K, use_crps=USE_CRPS_LOSS, use_focal=USE_FOCAL_LOSS,
            )

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total    += loss.item()
            sig_sum  += sig_val
            risk_sum += risk_val

    n = len(loader)
    return total / n, sig_sum / n, risk_sum / n


# ---------------------------------------------------------------------------
# Train a single ensemble member
# ---------------------------------------------------------------------------

def train_single(
    model_idx:     int,
    seed:          int,
    train_loader:  DataLoader,
    normal_loader: DataLoader,
    val_loader:    DataLoader,
    pos_weight:    float,
    device:        torch.device,
    K:             int,
) -> float:
    """Train one ensemble member and save it to MODEL_DIR/model_{model_idx}.pt.

    Curriculum learning: for the first CURRICULUM_EPOCHS epochs the model
    trains only on normal_loader (windows with no arrhythmia).  After that
    it switches to the full train_loader.

    Returns the best validation loss for this member.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    pw_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    model = ProbabilisticLSTM(
        input_size       = INPUT_CHANNELS,
        hidden_size      = HIDDEN_SIZE,
        num_layers       = NUM_LAYERS,
        forecast_len     = FORECAST_LEN,
        dropout          = DROPOUT,
        embed_dim        = EMBED_DIM,
        num_beat_classes = NUM_BEAT_CLASSES,
        use_attention    = USE_ATTENTION,
        use_rr_features  = USE_RR_FEATURES,
        use_layer_norm   = USE_LAYER_NORM,
        use_risk_head    = USE_RISK_HEAD,
        deterministic    = DETERMINISTIC,
        K                = K,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [member {model_idx}] Parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=LR_T0, T_mult=LR_T_MULT, eta_min=LEARNING_RATE * 0.01
    )

    ckpt_path = os.path.join(MODEL_DIR, f"model_{model_idx}.pt")
    stopper   = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, ckpt_path=ckpt_path)

    extra_ckpt = {
        "pos_weight": pos_weight,
        "temperature": 1.0,  # placeholder, updated after training
        "config": {
            "input_size":       INPUT_CHANNELS,
            "hidden_size":      HIDDEN_SIZE,
            "num_layers":       NUM_LAYERS,
            "forecast_len":     FORECAST_LEN,
            "dropout":          DROPOUT,
            "embed_dim":        EMBED_DIM,
            "num_beat_classes": NUM_BEAT_CLASSES,
            "use_attention":    USE_ATTENTION,
            "use_rr_features":  True,       # always True (4-channel features)
            "use_layer_norm":   USE_LAYER_NORM,
            "use_risk_head":    USE_RISK_HEAD,
            "deterministic":    DETERMINISTIC,
            "K":                K,
            "num_heads":        4,          # N_ATTN_HEADS
        },
    }

    history: dict[str, list] = {
        k: [] for k in
        ["train_total", "val_total", "train_sig", "val_sig", "train_risk", "val_risk"]
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        # Curriculum: use normal-only loader for early epochs
        active_loader = normal_loader if epoch <= CURRICULUM_EPOCHS else train_loader

        tr = run_epoch(model, active_loader, optimizer, device, pw_tensor,
                       training=True,  K=K)
        va = run_epoch(model, val_loader,    optimizer, device, pw_tensor,
                       training=False, K=K)

        for key, val in zip(
            ["train_total", "val_total", "train_sig", "val_sig", "train_risk", "val_risk"],
            [tr[0], va[0], tr[1], va[1], tr[2], va[2]],
        ):
            history[key].append(val)

        scheduler.step()

        stop = stopper.step(va[0], model, optimizer, epoch, extra_ckpt)
        curriculum_tag = " [curriculum]" if epoch <= CURRICULUM_EPOCHS else ""
        flag = "  << best" if stopper.improved else \
               f"  (patience {stopper.counter}/{stopper.patience})"

        print(
            f"  [m{model_idx} e{epoch:3d}/{NUM_EPOCHS}]{curriculum_tag}  "
            f"Total: {tr[0]:.4f}/{va[0]:.4f}  "
            f"Sig: {tr[1]:.4f}/{va[1]:.4f}  "
            f"Risk: {tr[2]:.4f}/{va[2]:.4f}  "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
            f"{flag}"
        )

        if stop:
            print(
                f"\n  [member {model_idx}] Early stopping at epoch {epoch} "
                f"(patience {EARLY_STOPPING_PATIENCE} exceeded)."
            )
            break

    # --- Fit temperature scaling on val set ---
    # Load best checkpoint, fit temperature, re-save with temperature added
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    ckpt["temperature"] = fit_temperature(model, val_loader, device)
    torch.save(ckpt, ckpt_path)

    # --- Loss curves for this member ---
    n_ep    = len(history["train_total"])
    ep_axis = range(1, n_ep + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, tr_key, va_key, title in zip(
        axes,
        ["train_total", "train_sig",  "train_risk"],
        ["val_total",   "val_sig",    "val_risk"],
        ["Total Loss",  "Signal Loss", "Risk Loss"],
    ):
        ax.plot(ep_axis, history[tr_key], label="Train")
        ax.plot(ep_axis, history[va_key], label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.suptitle(f"Training Progress — member {model_idx} (seed={seed})", fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, f"loss_curves_model_{model_idx}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(
        f"\n  [member {model_idx}] Done.  "
        f"Best val loss: {stopper.best_loss:.4f}  → {ckpt_path}"
    )
    return stopper.best_loss


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train() -> None:
    """Train N_ENSEMBLE independent models and save model_0.pt ... model_{N-1}.pt."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K      = K_MDN if USE_MDN else 1
    mode   = "deterministic (MSE)" if DETERMINISTIC else (
        f"MDN K={K}" if K > 1 else
        ("CRPS" if USE_CRPS_LOSS else f"Beta-NLL β={BETA_NLL}")
    )

    print(f"[train] Device      : {device}")
    print(f"[train] Mode        : {mode}")
    print(f"[train] Ensemble    : N={N_ENSEMBLE}")
    print(f"[train] Curriculum  : {CURRICULUM_EPOCHS} epochs on normal windows")
    print(
        f"[train] Features    : Attention={USE_ATTENTION}  RR(4ch)={USE_RR_FEATURES}  "
        f"LayerNorm={USE_LAYER_NORM}  RiskHead={USE_RISK_HEAD}  "
        f"FocalLoss={USE_FOCAL_LOSS}  Augment={AUGMENT_TRAIN}\n"
    )

    # --- Data ---
    # get_dataloaders returns 4-tuple: train_loader, val_loader, cal_loader, pos_weight
    train_loader, val_loader, cal_loader, pos_weight = get_dataloaders(
        data_folder=DATA_FOLDER, input_len=INPUT_LEN, forecast_len=FORECAST_LEN,
        stride=STRIDE, batch_size=BATCH_SIZE, split=TRAIN_VAL_SPLIT, seed=SEED,
    )

    # Build a normal-only loader for curriculum learning.
    # train_loader.dataset is the ECGDataset; Subset restricts to normal_indices.
    train_ds      = train_loader.dataset
    normal_subset = Subset(train_ds, train_ds.normal_indices)
    normal_loader = DataLoader(
        normal_subset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    print(
        f"[train] Normal-only subset: {len(normal_subset):,} windows "
        f"({100*len(normal_subset)/len(train_ds):.1f}% of train)\n"
    )

    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    best_losses = []
    for model_idx in range(N_ENSEMBLE):
        seed = SEED + model_idx
        print(f"\n{'='*60}")
        print(f"[train] Ensemble member {model_idx}/{N_ENSEMBLE-1}  seed={seed}")
        print(f"{'='*60}")
        best_val = train_single(
            model_idx=model_idx,
            seed=seed,
            train_loader=train_loader,
            normal_loader=normal_loader,
            val_loader=val_loader,
            pos_weight=pos_weight,
            device=device,
            K=K,
        )
        best_losses.append(best_val)

    print(f"\n{'='*60}")
    print(f"[train] Ensemble training complete.")
    for i, loss in enumerate(best_losses):
        print(f"  model_{i}.pt  best_val_loss={loss:.4f}")
    print(f"  Mean val loss : {np.mean(best_losses):.4f}")
    print(f"  Models saved to '{MODEL_DIR}/'")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()

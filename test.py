# test.py
# Evaluation script for the annotation-aware probabilistic ECG forecaster.
#
# Run:  python test.py
#
# What this script does
# ---------------------
# 1. Loads the best checkpoint from MODEL_DIR.
# 2. Evaluates on the validation set and reports:
#      Signal head  — MSE and Gaussian NLL
#      Risk head    — binary accuracy and AUC-ROC (no sklearn needed)
# 3. Produces three figures saved to RESULTS_DIR:
#      predictions.png         — forecast with ±1σ/±2σ uncertainty bands
#      uncertainty_horizon.png — aleatoric vs epistemic σ across forecast horizon
#      risk_analysis.png       — predicted arrhythmia risk vs ground truth
#
# GenAI assistance: used to draft the MC Dropout loop, AUC computation, and
# plot layout; the uncertainty decomposition was verified against Kendall &
# Gal (2017).

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
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    model = ProbabilisticLSTM(
        input_size      = cfg["input_size"],
        hidden_size     = cfg["hidden_size"],
        num_layers      = cfg["num_layers"],
        forecast_len    = cfg["forecast_len"],
        dropout         = cfg["dropout"],
        embed_dim       = cfg["embed_dim"],
        num_beat_classes= cfg["num_beat_classes"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(
        f"[test] Loaded checkpoint from epoch {ckpt['epoch']} "
        f"(val total loss: {ckpt['val_loss']:.4f})"
    )
    return model


# ---------------------------------------------------------------------------
# MC Dropout inference
# ---------------------------------------------------------------------------

def mc_dropout_predict(
    model:    ProbabilisticLSTM,
    x_signal: torch.Tensor,
    x_annot:  torch.Tensor,
    n_samples: int = MC_SAMPLES,
) -> tuple[torch.Tensor, ...]:
    """Run *n_samples* stochastic forward passes with dropout active.

    Returns
    -------
    combined_mean : (batch, forecast_len)  — mean of μ across samples
    total_std     : (batch, forecast_len)  — sqrt(epistemic + aleatoric var)
    epistemic_var : (batch, forecast_len)  — variance of μ across samples
    aleatoric_var : (batch, forecast_len)  — mean σ² across samples
    risk_prob     : (batch,)               — mean sigmoid(logit) across samples
    """
    model.train()   # keep dropout active
    all_means, all_vars, all_risks = [], [], []

    with torch.no_grad():
        for _ in range(n_samples):
            sig_out, risk_logit = model(x_signal, x_annot)
            all_means.append(sig_out[..., 0])
            all_vars.append(torch.exp(sig_out[..., 1]))
            all_risks.append(torch.sigmoid(risk_logit).squeeze(-1))

    all_means = torch.stack(all_means)   # (n, batch, forecast_len)
    all_vars  = torch.stack(all_vars)
    all_risks = torch.stack(all_risks)   # (n, batch)

    # Law of total variance for uncertainty decomposition
    aleatoric_var = all_vars.mean(dim=0)              # (batch, forecast_len)
    epistemic_var = all_means.var(dim=0)
    combined_mean = all_means.mean(dim=0)
    total_std     = torch.sqrt(aleatoric_var + epistemic_var)
    risk_prob     = all_risks.mean(dim=0)             # (batch,)

    return combined_mean, total_std, epistemic_var, aleatoric_var, risk_prob


# ---------------------------------------------------------------------------
# AUC-ROC (pure numpy, no sklearn)
# ---------------------------------------------------------------------------

def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC-ROC via trapezoidal integration (no sklearn required)."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    desc      = np.argsort(-y_score)
    y_sorted  = y_true[desc]
    nP = y_sorted.sum()
    nN = len(y_sorted) - nP
    if nP == 0 or nN == 0:
        return float("nan")
    tpr = np.cumsum(y_sorted)      / nP
    fpr = np.cumsum(1 - y_sorted)  / nN
    return float(np.trapz(tpr, fpr))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model:  ProbabilisticLSTM,
    loader,
    device: torch.device,
) -> dict:
    """Return a dict of scalar metrics over the full loader.

    Metrics
    -------
    mse      : mean squared error of predicted μ vs ground truth
    nll      : Gaussian NLL of the signal head
    risk_acc : binary accuracy of arrhythmia risk (threshold 0.5)
    risk_auc : AUC-ROC for arrhythmia risk probability
    """
    model.eval()
    total_nll = total_mse = 0.0
    n_batches = 0
    all_risk_probs, all_risk_labels = [], []

    with torch.no_grad():
        for x_sig, x_ann, y_sig, y_risk in loader:
            x_sig  = x_sig.to(device)
            x_ann  = x_ann.to(device)
            y_sig  = y_sig.to(device)
            y_risk = y_risk.to(device)

            sig_out, risk_logit = model(x_sig, x_ann)
            total_nll += gaussian_nll_loss(sig_out, y_sig).item()
            total_mse += ((sig_out[..., 0] - y_sig) ** 2).mean().item()

            risk_prob = torch.sigmoid(risk_logit).squeeze(-1)
            all_risk_probs.append(risk_prob.cpu().numpy())
            all_risk_labels.append(y_risk.cpu().numpy())
            n_batches += 1

    probs  = np.concatenate(all_risk_probs)
    labels = np.concatenate(all_risk_labels)
    preds  = (probs >= 0.5).astype(float)

    return {
        "mse":      total_mse / n_batches,
        "nll":      total_nll / n_batches,
        "risk_acc": float((preds == labels).mean()),
        "risk_auc": roc_auc(labels, probs),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_predictions(
    model:    ProbabilisticLSTM,
    loader,
    device:   torch.device,
    n_examples: int = 6,
) -> None:
    """Forecast examples with MC Dropout uncertainty bands.
    Title of each panel shows the ground-truth and predicted arrhythmia risk.
    Saved to RESULTS_DIR/predictions.png.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    x_sig, x_ann, y_sig, y_risk = next(iter(loader))
    x_sig, x_ann = x_sig.to(device), x_ann.to(device)

    mean, total_std, epi_var, ale_var, risk_prob = mc_dropout_predict(
        model, x_sig, x_ann
    )

    mean      = mean.cpu().numpy()
    total_std = total_std.cpu().numpy()
    epi_std   = np.sqrt(epi_var.cpu().numpy())
    ale_std   = np.sqrt(ale_var.cpu().numpy())
    x_np      = x_sig.cpu().squeeze(-1).numpy()
    y_np      = y_sig.numpy()
    risk_prob = risk_prob.cpu().numpy()
    y_risk    = y_risk.numpy()

    n = min(n_examples, len(x_np))
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]

    input_t    = np.arange(INPUT_LEN)
    forecast_t = np.arange(INPUT_LEN, INPUT_LEN + FORECAST_LEN)

    for i, ax in enumerate(axes):
        ax.plot(input_t,    x_np[i],  color="steelblue", lw=0.8, alpha=0.9,
                label="Input ECG")
        ax.plot(forecast_t, y_np[i],  color="green",     lw=1.2,
                label="Ground Truth")
        ax.plot(forecast_t, mean[i],  color="crimson",   lw=1.2, ls="--",
                label="Predicted μ")
        ax.fill_between(
            forecast_t,
            mean[i] - total_std[i], mean[i] + total_std[i],
            alpha=0.30, color="crimson", label="±1σ total",
        )
        ax.fill_between(
            forecast_t,
            mean[i] - 2 * total_std[i], mean[i] + 2 * total_std[i],
            alpha=0.12, color="crimson", label="±2σ total",
        )
        ax.axvline(INPUT_LEN, color="gray", ls=":", lw=1)

        true_risk_str = "YES" if y_risk[i] > 0.5 else "NO"
        pred_risk_str = f"{risk_prob[i]:.2f}"
        ax.set_title(
            f"Sample {i+1}  |  Arrhythmia in window — True: {true_risk_str}  "
            f"Predicted prob: {pred_risk_str}  |  "
            f"Ale σ: {ale_std[i].mean():.4f}  Epi σ: {epi_std[i].mean():.4f}"
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
    print(f"[test] Predictions plot      → {out}")


def plot_uncertainty_horizon(
    model:  ProbabilisticLSTM,
    loader,
    device: torch.device,
    max_batches: int = 40,
) -> None:
    """Aleatoric vs epistemic σ across the forecast horizon.
    Saved to RESULTS_DIR/uncertainty_horizon.png.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_ale, all_epi = [], []

    for i, (x_sig, x_ann, _, _) in enumerate(loader):
        if i >= max_batches:
            break
        x_sig, x_ann = x_sig.to(device), x_ann.to(device)
        _, _, epi_var, ale_var, _ = mc_dropout_predict(model, x_sig, x_ann, n_samples=20)
        all_ale.append(ale_var.cpu().numpy())
        all_epi.append(epi_var.cpu().numpy())

    ale = np.sqrt(np.concatenate(all_ale).mean(axis=0))  # (forecast_len,)
    epi = np.sqrt(np.concatenate(all_epi).mean(axis=0))
    t   = np.arange(FORECAST_LEN) / 360                  # seconds

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
    print(f"[test] Uncertainty horizon   → {out}")


def plot_risk_analysis(
    model:  ProbabilisticLSTM,
    loader,
    device: torch.device,
    max_batches: int = 60,
) -> None:
    """Visualise predicted arrhythmia risk probability vs ground-truth labels.

    Produces two panels:
      Left  — distribution of predicted probabilities for positive/negative windows
      Right — ROC curve

    Saved to RESULTS_DIR/risk_analysis.png.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_probs, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for i, (x_sig, x_ann, _, y_risk) in enumerate(loader):
            if i >= max_batches:
                break
            x_sig, x_ann = x_sig.to(device), x_ann.to(device)
            _, risk_logit = model(x_sig, x_ann)
            probs = torch.sigmoid(risk_logit).squeeze(-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y_risk.numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]

    # --- ROC curve ---
    desc     = np.argsort(-probs)
    y_sorted = labels[desc]
    nP, nN   = y_sorted.sum(), (1 - y_sorted).sum()
    tpr = np.concatenate([[0], np.cumsum(y_sorted)      / max(nP, 1), [1]])
    fpr = np.concatenate([[0], np.cumsum(1 - y_sorted)  / max(nN, 1), [1]])
    auc = float(np.trapz(tpr, fpr))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1 — predicted probability distributions
    bins = np.linspace(0, 1, 30)
    ax1.hist(neg_probs, bins=bins, alpha=0.6, color="steelblue",
             label=f"No arrhythmia (n={len(neg_probs):,})", density=True)
    ax1.hist(pos_probs, bins=bins, alpha=0.6, color="crimson",
             label=f"Arrhythmia (n={len(pos_probs):,})", density=True)
    ax1.axvline(0.5, color="black", ls="--", lw=1, label="Decision threshold")
    ax1.set_xlabel("Predicted Risk Probability")
    ax1.set_ylabel("Density")
    ax1.set_title("Risk Score Distribution by Class")
    ax1.legend(fontsize=9)

    # Panel 2 — ROC curve
    ax2.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.3f}")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve — Arrhythmia Risk Prediction")
    ax2.legend()

    plt.suptitle("Arrhythmia Risk Head Evaluation", fontsize=13)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "risk_analysis.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[test] Risk analysis plot    → {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(MODEL_DIR, "best_model.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint found at '{ckpt_path}'. "
            "Please run  python train.py  first."
        )

    model = load_model(ckpt_path, device)

    _, val_loader, _ = get_dataloaders(
        data_folder=DATA_FOLDER,
        input_len=INPUT_LEN,
        forecast_len=FORECAST_LEN,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
        split=TRAIN_VAL_SPLIT,
        seed=SEED,
    )

    # --- Quantitative metrics ---
    metrics = evaluate(model, val_loader, device)
    print(f"\n[test] ── Evaluation Results ────────────────────────────")
    print(f"[test]   Signal MSE                   : {metrics['mse']:.6f}")
    print(f"[test]   Signal NLL (Gaussian)         : {metrics['nll']:.6f}")
    print(f"[test]   Arrhythmia Risk Accuracy      : {metrics['risk_acc']:.4f}")
    print(f"[test]   Arrhythmia Risk AUC-ROC       : {metrics['risk_auc']:.4f}")
    print(f"[test] ─────────────────────────────────────────────────\n")

    # --- Figures ---
    plot_predictions(model, val_loader, device, n_examples=6)
    plot_uncertainty_horizon(model, val_loader, device)
    plot_risk_analysis(model, val_loader, device)

    print(f"\n[test] All results saved to '{RESULTS_DIR}/'")


if __name__ == "__main__":
    main()

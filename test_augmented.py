# test.py
# Evaluation script for the annotation-aware probabilistic ECG forecaster.
#
# Run:  python test.py
#
# Improvements over baseline
# --------------------------
#   Deep Ensemble + MC Dropout : combines N_ENSEMBLE models × MC_SAMPLES passes
#                                for richer epistemic uncertainty.
#   MDN support                : works with K=1 (single Gaussian) or K>1 (MDN).
#   Conformal prediction       : normalized residual quantile on cal_loader gives
#                                distribution-free coverage guarantee.
#   Conformal risk prediction  : nonconformity-score set for risk head.
#   Winkler score              : interval sharpness metric at alpha=0.10.
#   Per-beat-type metrics      : MSE stratified by beat type (Normal/PVC/APB/Other).
#   CRPS metric                : closed-form proper scoring rule.
#   Calibration diagram        : reliability plot with conformal level marked.
#   Arrhythmia uncertainty     : aleatoric vs epistemic σ by arrhythmia presence.
#   Attention heatmap          : which input timesteps the model focuses on.
#   Temperature scaling        : calibrated risk logit division per model.
#   HRV features               : SDNN, RMSSD, pNN50 passed to all model calls.
#   Diebold-Mariano test       : statistical test for ensemble vs single model.
#
# Figures produced
# ----------------
#   predictions.png                     — forecast with uncertainty bands + conformal interval
#   uncertainty_horizon.png             — aleatoric vs epistemic σ across forecast horizon
#   risk_analysis.png                   — risk score distribution + ROC curve
#   calibration.png                     — reliability diagram + conformal coverage level
#   arrhythmia_uncertainty.png          — uncertainty stratified by arrhythmia presence
#   attention_weights.png               — attention heatmap over input ECG (if applicable)
#   uncertainty_scatter.png             — scatter plot of prediction error vs uncertainty with correlation
#   error_vs_uncertainty_quantile.png   — mean prediction error across uncertainty quantiles (low → high)
#   retention_curve.png                 — error as a function of retained samples sorted by uncertainty
#
# GenAI assistance: used to draft the MC Dropout loop, CRPS formula, AUC
# computation, calibration diagram, conformal prediction logic, and Winkler
# score; verified against Kendall & Gal (2017), Gneiting & Raftery (2007),
# and Angelopoulos & Bates (2022). Extended with temperature scaling, HRV
# feature passing, conformal risk prediction, and Diebold-Mariano test.

import os
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import (
    DATA_FOLDER, INPUT_LEN, FORECAST_LEN, STRIDE,
    BATCH_SIZE, TRAIN_VAL_SPLIT, SEED,
    MODEL_DIR, RESULTS_DIR, MC_SAMPLES, N_ENSEMBLE,
    USE_MDN, K_MDN, CONFORMAL_ALPHA_RISK,
)
from dataset_augmented import get_augmented_dataloaders, WINDOW_SUBTYPES
from model import ProbabilisticLSTM, gaussian_nll_loss, gaussian_crps, signal_mean, signal_variance


# ---------------------------------------------------------------------------
# Model loading (ensemble)
# ---------------------------------------------------------------------------

def load_ensemble(
    model_dir: str,
    device:    torch.device,
) -> list[ProbabilisticLSTM]:
    """Load all available model_*.pt checkpoints from model_dir.

    Tries model_0.pt, model_1.pt, ... and stops when a file is not found.
    Falls back to best_model.pt if no model_*.pt files exist (legacy compat).

    Returns a list of models (all in eval mode by default).
    """
    models = []
    idx    = 0
    while True:
        path = os.path.join(model_dir, f"model_{idx}.pt")
        if not os.path.exists(path):
            break
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        cfg   = ckpt["config"]
        model = ProbabilisticLSTM(**cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        # Load temperature for calibrated risk prediction
        model.temperature = ckpt.get("temperature", 1.0)
        print(
            f"[test] Loaded model_{idx}.pt  epoch={ckpt['epoch']}  "
            f"val_loss={ckpt['val_loss']:.4f}  K={cfg.get('K', 1)}  "
            f"T={model.temperature:.4f}"
        )
        models.append(model)
        idx += 1

    if not models:
        # Legacy fallback
        legacy = os.path.join(model_dir, "best_model.pt")
        if os.path.exists(legacy):
            ckpt  = torch.load(legacy, map_location=device, weights_only=False)
            cfg   = ckpt["config"]
            model = ProbabilisticLSTM(**cfg).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            model.temperature = ckpt.get("temperature", 1.0)
            print(
                f"[test] Loaded best_model.pt (legacy)  "
                f"epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}"
            )
            models.append(model)

    if not models:
        raise FileNotFoundError(
            f"No checkpoints found in '{model_dir}'. Run python train.py first."
        )

    print(f"[test] Ensemble size: {len(models)} model(s)\n")
    return models


# ---------------------------------------------------------------------------
# Ensemble + MC Dropout inference
# ---------------------------------------------------------------------------

def ensemble_mc_predict(
    models:  list[ProbabilisticLSTM],
    x_sig:   torch.Tensor,
    x_ann:   torch.Tensor,
    x_feat:  torch.Tensor,
    K:       int,
    x_hrv:   torch.Tensor | None = None,
    x_feat_mask:  torch.Tensor | None = None,
    x_beat_event: torch.Tensor | None = None,
    n_mc:    int = MC_SAMPLES,
) -> tuple:
    """Combined ensemble + MC Dropout inference.

    For each model in the ensemble, run n_mc stochastic forward passes
    (dropout active).  Aggregate via the law of total variance:

        aleatoric_var  = mean over all (model × MC) runs of predicted σ²
        epistemic_var  = variance of predicted means across ensemble members
        total_std      = sqrt(aleatoric_var + epistemic_var)

    Parameters
    ----------
    models       : list of ProbabilisticLSTM (loaded ensemble)
    x_sig        : (batch, input_len, 2)
    x_ann        : (batch, input_len)
    x_feat       : (batch, input_len, 4)
    K            : number of mixture components (from checkpoint config)
    x_hrv        : (batch, 3) or None
    x_feat_mask  : (batch, input_len, 4) or None 
    x_beat_event : (batch, input_len, 7) or None
    n_mc         : MC Dropout samples per ensemble member

    Returns
    -------
    combined_mean : (batch, forecast_len)
    total_std     : (batch, forecast_len)
    epistemic_var : (batch, forecast_len)
    aleatoric_var : (batch, forecast_len)
    risk_prob     : (batch,) or None
    """
    member_means = []
    member_vars  = []
    all_risks    = []

    for model in models:
        model.train()   # keep dropout stochastic
        mc_means = []
        mc_vars  = []
        mc_risks = []

        with torch.no_grad():
            for _ in range(n_mc):
                sig_out, risk_logit = model(x_sig, x_ann, x_feat, x_hrv=x_hrv,
                                            x_feat_mask=x_feat_mask, x_beat_event=x_beat_event)
                mu    = signal_mean(sig_out, K=K)
                var   = signal_variance(sig_out, K=K)
                mc_means.append(mu)
                mc_vars.append(var)
                if risk_logit is not None:
                    risk_logit_cal = risk_logit / getattr(model, 'temperature', 1.0)
                    mc_risks.append(torch.sigmoid(risk_logit_cal).squeeze(-1))

        # Average MC passes for this member
        mc_means_t = torch.stack(mc_means)   # (n_mc, batch, T)
        mc_vars_t  = torch.stack(mc_vars)    # (n_mc, batch, T)
        member_means.append(mc_means_t.mean(dim=0))     # (batch, T)
        member_vars.append(mc_vars_t.mean(dim=0))       # (batch, T)
        if mc_risks:
            all_risks.append(torch.stack(mc_risks).mean(dim=0))

    member_means_t = torch.stack(member_means)   # (N, batch, T)
    member_vars_t  = torch.stack(member_vars)    # (N, batch, T)

    combined_mean = member_means_t.mean(dim=0)               # (batch, T)
    aleatoric_var = member_vars_t.mean(dim=0)                 # E[sigma^2]
    epistemic_var = member_means_t.var(dim=0)                 # Var[mu]
    total_std     = torch.sqrt(aleatoric_var + epistemic_var)

    risk_prob = torch.stack(all_risks).mean(dim=0) if all_risks else None

    return combined_mean, total_std, epistemic_var, aleatoric_var, risk_prob


# ---------------------------------------------------------------------------
# Conformal prediction calibration (signal)
# ---------------------------------------------------------------------------

def conformal_calibrate(
    models:     list[ProbabilisticLSTM],
    cal_loader,
    device:     torch.device,
    K:          int,
    alpha:      float = 0.10,
) -> float:
    """Compute normalized conformal quantile on the calibration set.

    Runs the ensemble in eval mode (no MC Dropout) and collects normalized
    (studentized) residuals r_i = |y_i - mu_i| / sigma_i for every timestep.

    Returns q_conformal such that the prediction interval
        [mu - q * sigma, mu + q * sigma]
    achieves at least (1-alpha) marginal coverage on held-out data.

    Uses the standard conformal guarantee:
        q = quantile(residuals, ceil((n+1)*(1-alpha)) / n)
    """
    residuals = []

    for model in models:
        model.eval()

    with torch.no_grad():
        for batch in cal_loader:
            x_sig        = batch["x_signal"].to(device)
            x_ann        = batch["x_annot"].to(device)
            x_feat       = batch["x_feat"].to(device)
            x_hrv        = batch["x_hrv"].to(device)
            y_sig        = batch["y_signal"].to(device)
            x_feat_mask  = batch["x_feat_mask"].to(device)
            x_beat_event = batch["x_beat_event"].to(device)

            # Average ensemble predictions (single pass, no MC)
            batch_means = []
            batch_vars  = []
            for model in models:
                sig_out, _ = model(x_sig, x_ann, x_feat, x_hrv=x_hrv,
                                   x_feat_mask=x_feat_mask, x_beat_event=x_beat_event)
                batch_means.append(signal_mean(sig_out, K=K))
                batch_vars.append(signal_variance(sig_out, K=K))

            mu    = torch.stack(batch_means).mean(dim=0)      # (batch, T)
            var   = torch.stack(batch_vars).mean(dim=0)       # (batch, T)
            sigma = torch.sqrt(var).clamp(min=1e-6)

            # Normalized residual |y - mu| / sigma, flattened to scalar per timestep
            res = ((y_sig - mu).abs() / sigma).cpu().numpy()  # (batch, T)
            residuals.append(res.ravel())

    residuals = np.concatenate(residuals)
    n         = len(residuals)
    # Conformal quantile level: ceil((n+1)*(1-alpha)) / n, clipped to [0,1]
    level = min(math.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    q     = float(np.quantile(residuals, level))
    print(
        f"[test] Conformal calibration: n={n:,}  alpha={alpha}  "
        f"level={level:.6f}  q={q:.4f}"
    )
    return q


# ---------------------------------------------------------------------------
# Conformal prediction calibration (risk head)
# ---------------------------------------------------------------------------

def conformal_calibrate_risk(
    models: list,
    cal_loader,
    device:  torch.device,
    alpha:   float = 0.10,
) -> float:
    """Nonconformity scores for risk head. Returns conformal threshold q_risk.

    Score for sample i: 1 - p(y_true_i | x_i)
      y=0: score = prob  (= 1 - p[0])
      y=1: score = 1 - prob  (= 1 - p[1])

    Prediction set at test time: include class c if score(c) <= q_risk.
      Include 0 if prob    <= q_risk
      Include 1 if 1-prob  <= q_risk  i.e. prob >= 1-q_risk
    """
    scores = []
    for model in models:
        model.eval()
    with torch.no_grad():
        for batch in cal_loader:
            x_sig        = batch["x_signal"].to(device)
            x_ann        = batch["x_annot"].to(device)
            x_feat       = batch["x_feat"].to(device)
            x_hrv        = batch["x_hrv"].to(device)
            y_risk       = batch["y_risk"].numpy()
            x_feat_mask  = batch["x_feat_mask"].to(device)
            x_beat_event = batch["x_beat_event"].to(device)
            batch_probs = []
            for model in models:
                _, risk_logit = model(x_sig, x_ann, x_feat, x_hrv=x_hrv,
                                      x_feat_mask=x_feat_mask, x_beat_event=x_beat_event)
                if risk_logit is None:
                    return float("nan")
                logit_cal = risk_logit / getattr(model, 'temperature', 1.0)
                batch_probs.append(torch.sigmoid(logit_cal).squeeze(-1))
            prob = torch.stack(batch_probs).mean(dim=0).cpu().numpy()
            # nonconformity score: 1 - p(true class)
            nc_score = np.where(y_risk > 0.5, 1.0 - prob, prob)
            scores.append(nc_score)

    scores = np.concatenate(scores)
    n = len(scores)
    level = min(math.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    q_risk = float(np.quantile(scores, level))
    print(
        f"[test] Conformal risk calibration: n={n:,}  alpha={alpha}  "
        f"level={level:.6f}  q_risk={q_risk:.4f}"
    )
    return q_risk


# ---------------------------------------------------------------------------
# Winkler score
# ---------------------------------------------------------------------------

def winkler_score(
    mu:          np.ndarray,
    sigma:       np.ndarray,
    y:           np.ndarray,
    q_conformal: float,
    alpha:       float = 0.10,
) -> float:
    """Compute the mean Winkler Score for a prediction interval.

    The interval is [lower, upper] = [mu - q*sigma, mu + q*sigma].

    WS_i = (upper_i - lower_i)
           + (2/alpha) * max(0, lower_i - y_i)   [lower violation]
           + (2/alpha) * max(0, y_i - upper_i)   [upper violation]

    Lower WS = narrower intervals with fewer violations = better.

    Parameters
    ----------
    mu, sigma, y : numpy arrays of the same shape (any shape, computed flat)
    q_conformal  : conformal quantile (multiplied with sigma to get half-width)
    alpha        : nominal miscoverage rate (default 0.10 → 90% interval)

    Returns
    -------
    float : mean Winkler Score over all elements
    """
    lower = mu - q_conformal * sigma
    upper = mu + q_conformal * sigma
    width = upper - lower                                  # always positive
    pen_l = np.maximum(0.0, lower - y) * (2.0 / alpha)
    pen_u = np.maximum(0.0, y - upper) * (2.0 / alpha)
    return float((width + pen_l + pen_u).mean())


# ---------------------------------------------------------------------------
# Optimal classification threshold (Youden's J index)
# ---------------------------------------------------------------------------

def find_optimal_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """Find the decision threshold that maximises Youden's J index.

    Youden's J = sensitivity + specificity − 1  =  TPR + TNR − 1  =  TPR − FPR

    Scans 199 candidate thresholds in [0.01, 0.99] and picks the one that
    maximises J.  Equivalent to finding the point on the ROC curve farthest
    above the diagonal.

    Parameters
    ----------
    probs  : (n,) predicted probability of positive class
    labels : (n,) binary ground-truth labels {0, 1}

    Returns
    -------
    float : optimal threshold t* ∈ [0.01, 0.99]
    """
    nP = float(labels.sum())
    nN = float(len(labels) - nP)
    if nP == 0 or nN == 0:
        return 0.5

    best_j, best_t = -1.0, 0.5
    for t in np.linspace(0.01, 0.99, 199):
        pred = (probs >= t).astype(float)
        tpr  = (pred * labels).sum() / nP
        tnr  = ((1 - pred) * (1 - labels)).sum() / nN
        j    = tpr + tnr - 1.0
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def f1_at_threshold(probs: np.ndarray, labels: np.ndarray, t: float) -> tuple[float, float, float]:
    """Return (precision, recall, F1) at a given threshold."""
    pred = (probs >= t).astype(float)
    tp   = (pred * labels).sum()
    fp   = (pred * (1 - labels)).sum()
    fn   = ((1 - pred) * labels).sum()
    prec = tp / max(tp + fp, 1e-9)
    rec  = tp / max(tp + fn, 1e-9)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    return float(prec), float(rec), float(f1)


# ---------------------------------------------------------------------------
# AUC-ROC (pure numpy)
# ---------------------------------------------------------------------------

def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true  = np.asarray(y_true,  dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    desc    = np.argsort(-y_score)
    ys      = y_true[desc]
    nP, nN  = ys.sum(), len(ys) - ys.sum()
    if nP == 0 or nN == 0:
        return float("nan")
    tpr = np.cumsum(ys)       / nP
    fpr = np.cumsum(1 - ys)   / nN
    return float(np.trapezoid(tpr, fpr))


# ---------------------------------------------------------------------------
# Diebold-Mariano test
# ---------------------------------------------------------------------------

def diebold_mariano_test(errors1: np.ndarray, errors2: np.ndarray) -> tuple[float, float]:
    """Two-sided Diebold-Mariano test for equal predictive accuracy.

    Tests H0: E[L(e1)] = E[L(e2)]  where L = squared error.
    Uses the Harvey-Leybourne-Newbold (1997) small-sample correction.

    Returns (dm_statistic, p_value).
    """
    d  = errors1 ** 2 - errors2 ** 2      # loss differential
    n  = len(d)
    d_bar  = d.mean()
    d_var  = ((d - d_bar) ** 2).sum() / (n * (n - 1))  # variance of mean
    if d_var <= 0:
        return 0.0, 1.0
    dm = d_bar / math.sqrt(d_var)
    # Two-sided p-value via standard normal
    p  = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(dm) / math.sqrt(2))))
    return float(dm), float(p)


def collect_forecast_errors(models_a, models_b, loader, device, K, max_batches=60):
    """Return (errors_a, errors_b) as flat arrays of per-sample mean squared errors."""
    def _get_errors(models, loader):
        errs = []
        for m in models: m.eval()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= max_batches: 
                    break
                x_sig        = batch["x_signal"].to(device)
                x_ann        = batch["x_annot"].to(device)
                x_feat       = batch["x_feat"].to(device)
                y_sig        = batch["y_signal"].to(device)
                x_hrv        = batch["x_hrv"].to(device)
                x_feat_mask  = batch["x_feat_mask"].to(device)
                x_beat_event = batch["x_beat_event"].to(device)
                means = []
                for m in models:
                    sig_out, _ = m(x_sig, x_ann, x_feat, x_hrv=x_hrv,
                                   x_feat_mask=x_feat_mask, x_beat_event=x_beat_event)
                    means.append(signal_mean(sig_out, K=K))
                mu = torch.stack(means).mean(0)
                errs.append(((mu - y_sig.to(device))**2).mean(-1).cpu().numpy())
        return np.concatenate(errs)
    return _get_errors(models_a, loader), _get_errors(models_b, loader)


# ---------------------------------------------------------------------------
# Quantitative evaluation
# ---------------------------------------------------------------------------

def evaluate(
    models:      list[ProbabilisticLSTM],
    loader,
    device:      torch.device,
    K:           int,
    q_conformal: float | None = None,
) -> dict:
    """Return evaluation metrics over a data loader using ensemble mean predictions.

    Each model is run in eval mode with a single forward pass (no MC Dropout).
    Predictions are averaged over ensemble members.

    Metrics
    -------
    mse      : mean squared error of predicted μ vs ground truth
    nll      : Gaussian NLL (nan for MDN or deterministic; uses first component mean)
    crps     : Continuous Ranked Probability Score (nan if not applicable)
    risk_acc : binary accuracy at threshold 0.5
    risk_auc : AUC-ROC for risk probability
    winkler  : Winkler Score at alpha=0.10 (only if q_conformal is provided)
    """
    for model in models:
        model.eval()

    total_nll = total_mse = total_crps = 0.0
    n_nll     = 0
    n_batches = 0
    all_probs, all_labels = [], []
    all_mu, all_sigma, all_y = [], [], []
    all_errors = []
    all_uncertainties = []
    all_ale_unc   = []
    all_epi_unc   = []

    with torch.no_grad():
        for batch in loader:
            x_sig        = batch["x_signal"].to(device)
            x_ann        = batch["x_annot"].to(device)
            x_feat       = batch["x_feat"].to(device)
            y_sig        = batch["y_signal"].to(device)
            y_risk       = batch["y_risk"].to(device)
            x_hrv        = batch["x_hrv"].to(device)
            x_feat_mask  = batch["x_feat_mask"].to(device)
            x_beat_event = batch["x_beat_event"].to(device)

            # Average over ensemble members
            batch_means  = []
            batch_vars   = []
            batch_risks  = []
            batch_sig_out_k1 = []   # keep first K=1-equivalent output for CRPS/NLL

            for model in models:
                sig_out, risk_logit = model(x_sig, x_ann, x_feat, x_hrv=x_hrv,
                                            x_feat_mask=x_feat_mask, x_beat_event=x_beat_event)
                batch_means.append(signal_mean(sig_out, K=K))
                batch_vars.append(signal_variance(sig_out, K=K))
                if K == 1:
                    batch_sig_out_k1.append(sig_out)
                if risk_logit is not None:
                    risk_logit_cal = risk_logit / getattr(model, 'temperature', 1.0)
                    batch_risks.append(torch.sigmoid(risk_logit_cal).squeeze(-1))

            mu  = torch.stack(batch_means).mean(dim=0)   # (batch, T)
            var = torch.stack(batch_vars).mean(dim=0)    # (batch, T)

            total_mse += ((mu - y_sig) ** 2).mean().item()

            # Compute total error and uncertainties
            error = ((mu - y_sig) ** 2).mean(dim=-1)   # per sample
            uncertainty = torch.sqrt(var).mean(dim=-1)
            ale_unc   = torch.sqrt(torch.stack(batch_vars).mean(dim=0)).mean(dim=-1)
            epi_unc   = torch.sqrt(torch.stack(batch_means).var(dim=0)).mean(dim=-1)

            all_errors.append(error.cpu().numpy())
            all_uncertainties.append(uncertainty.cpu().numpy())
            all_ale_unc.append(ale_unc.cpu().numpy())
            all_epi_unc.append(epi_unc.cpu().numpy())

            # NLL and CRPS only for single-Gaussian (K=1) probabilistic mode
            if K == 1 and batch_sig_out_k1:
                avg_sig_out = torch.stack(batch_sig_out_k1).mean(dim=0)
                if avg_sig_out.shape[-1] == 2:
                    total_nll  += gaussian_nll_loss(avg_sig_out, y_sig, beta=0.0).item()
                    total_crps += gaussian_crps(avg_sig_out, y_sig)
                    n_nll      += 1

            if batch_risks:
                prob = torch.stack(batch_risks).mean(dim=0)
                all_probs.append(prob.cpu().numpy())
                all_labels.append(y_risk.cpu().numpy())

            if q_conformal is not None:
                all_mu.append(mu.cpu().numpy())
                all_sigma.append(torch.sqrt(var).clamp(min=1e-6).cpu().numpy())
                all_y.append(y_sig.cpu().numpy())

            n_batches += 1

    probs  = np.concatenate(all_probs)  if all_probs  else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    errors = np.concatenate(all_errors)
    uncs   = np.concatenate(all_uncertainties)

    corr = np.corrcoef(errors, uncs)[0,1]
    print(f"[test] Error–Uncertainty correlation: {corr:.4f}")
    
    # --- Threshold selection ---
    if len(probs):
        opt_t          = find_optimal_threshold(probs, labels)
        prec, rec, f1  = f1_at_threshold(probs, labels, opt_t)
        _, _, f1_half  = f1_at_threshold(probs, labels, 0.5)
        acc_opt        = float(((probs >= opt_t) == labels).mean())
        acc_half       = float(((probs >= 0.5)   == labels).mean())
    else:
        opt_t = 0.5
        prec = rec = f1 = f1_half = acc_opt = acc_half = float("nan")

    result = {
        "mse":          total_mse  / n_batches,
        "nll":          total_nll  / n_nll    if n_nll   else float("nan"),
        "crps":         total_crps / n_nll    if n_nll   else float("nan"),
        # fixed-threshold metrics
        "risk_acc":     acc_half,
        "risk_f1_half": f1_half,
        # Youden-optimal threshold metrics
        "opt_thresh":   opt_t,
        "risk_acc_opt": acc_opt,
        "risk_prec":    prec,
        "risk_recall":  rec,
        "risk_f1":      f1,
        # AUC (threshold-free)
        "risk_auc":     roc_auc(labels, probs) if len(probs) else float("nan"),
        "winkler":      float("nan"),
        "all_errors": np.concatenate(all_errors),
        "all_total_unc": np.concatenate(all_uncertainties),
        "all_ale_unc":   np.concatenate(all_ale_unc),
        "all_epi_unc":   np.concatenate(all_epi_unc),
    }

    if q_conformal is not None and all_mu:
        mu_arr    = np.concatenate(all_mu).ravel()
        sigma_arr = np.concatenate(all_sigma).ravel()
        y_arr     = np.concatenate(all_y).ravel()
        result["winkler"] = winkler_score(mu_arr, sigma_arr, y_arr, q_conformal)

    return result


# ---------------------------------------------------------------------------
# Per-beat-type evaluation
# ---------------------------------------------------------------------------

BEAT_TYPE_NAMES = {0: "No-beat", 1: "Normal", 2: "PVC", 3: "APB", 4: "Other"}


def per_beat_type_evaluate(
    models: list[ProbabilisticLSTM],
    loader,
    device: torch.device,
    K:      int,
) -> None:
    """Compute mean MSE stratified by dominant beat type in the forecast window.

    Beat types (y_beat_type, 6th element of batch):
        0 = no beat, 1 = normal, 2 = PVC, 3 = APB, 4 = other abnormal

    Prints a table of MSE per beat type.
    """
    for model in models:
        model.eval()

    mse_by_type: dict[int, list] = {t: [] for t in range(5)}

    with torch.no_grad():
        for batch in loader:
            x_sig        = batch["x_signal"].to(device)
            x_ann        = batch["x_annot"].to(device)
            x_feat       = batch["x_feat"].to(device)
            y_sig        = batch["y_signal"].to(device)
            x_hrv        = batch["x_hrv"].to(device)
            y_beat_type  = batch["y_beat_type"]
            x_feat_mask  = batch["x_feat_mask"].to(device)
            x_beat_event = batch["x_beat_event"].to(device)

            # Ensemble mean
            batch_means = []
            for model in models:
                sig_out, _ = model(x_sig, x_ann, x_feat, x_hrv=x_hrv,
                                   x_feat_mask=x_feat_mask, x_beat_event=x_beat_event)
                batch_means.append(signal_mean(sig_out, K=K))
            mu = torch.stack(batch_means).mean(dim=0)   # (batch, T)

            # Per-sample MSE
            mse_per_sample = ((mu - y_sig) ** 2).mean(dim=-1).cpu().numpy()  # (batch,)
            types          = y_beat_type.numpy()

            for t in range(5):
                mask = types == t
                if mask.any():
                    mse_by_type[t].extend(mse_per_sample[mask].tolist())

    print("\n[test] ── Per-Beat-Type MSE ──────────────────────────────────")
    print(f"  {'Beat Type':<18}  {'N windows':>10}  {'Mean MSE':>12}")
    print(f"  {'-'*18}  {'-'*10}  {'-'*12}")
    for t in range(5):
        vals = mse_by_type[t]
        if vals:
            print(
                f"  {BEAT_TYPE_NAMES[t]:<18}  {len(vals):>10,}  "
                f"{np.mean(vals):>12.6f}"
            )
        else:
            print(f"  {BEAT_TYPE_NAMES[t]:<18}  {'0':>10}  {'N/A':>12}")
    print("[test] ─────────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Plot 1 — Predictions with uncertainty bands
# ---------------------------------------------------------------------------

def plot_predictions(
    models:      list[ProbabilisticLSTM],
    loader,
    device:      torch.device,
    K:           int,
    q_conformal: float | None = None,
    n_examples:  int = 6,
) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    batch        = next(iter(loader))
    x_sig        = batch["x_signal"]
    x_ann        = batch["x_annot"]
    x_feat       = batch["x_feat"]
    y_sig        = batch["y_signal"]
    y_risk       = batch["y_risk"]
    x_hrv        = batch["x_hrv"]
    x_feat_mask  = batch["x_feat_mask"]
    x_beat_event = batch["x_beat_event"]

    mean, total_std, epi_var, ale_var, risk_prob = ensemble_mc_predict(
        models, x_sig.to(device), x_ann.to(device), x_feat.to(device), 
        K=K, x_hrv=x_hrv.to(device), x_feat_mask=x_feat_mask.to(device), 
        x_beat_event=x_beat_event.to(device), n_mc=20
    )
    mean      = mean.cpu().numpy()
    total_std = total_std.cpu().numpy()
    epi_std   = np.sqrt(epi_var.cpu().numpy())
    ale_std   = np.sqrt(ale_var.cpu().numpy())
    # x_sig is (batch, input_len, 2) — use lead 0 for display
    x_np      = x_sig.cpu()[:, :, 0].numpy()
    y_np      = y_sig.numpy()
    r_prob    = risk_prob.cpu().numpy() if risk_prob is not None else None
    r_true    = y_risk.numpy()

    n = min(n_examples, len(x_np))
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]

    t_in = np.arange(INPUT_LEN)
    t_fc = np.arange(INPUT_LEN, INPUT_LEN + FORECAST_LEN)

    for i, ax in enumerate(axes):
        ax.plot(t_in, x_np[i], color="steelblue", lw=0.8, alpha=0.9, label="Input ECG")
        ax.plot(t_fc, y_np[i], color="green",     lw=1.2,             label="Ground Truth")
        ax.plot(t_fc, mean[i], color="crimson",   lw=1.2, ls="--",    label="Predicted μ")
        ax.fill_between(t_fc, mean[i]-total_std[i], mean[i]+total_std[i],
                        alpha=0.30, color="crimson", label="±1σ total")
        ax.fill_between(t_fc, mean[i]-2*total_std[i], mean[i]+2*total_std[i],
                        alpha=0.12, color="crimson", label="±2σ total")
        # Conformal interval (dashed lines)
        if q_conformal is not None:
            conf_lo = mean[i] - q_conformal * total_std[i]
            conf_hi = mean[i] + q_conformal * total_std[i]
            ax.plot(t_fc, conf_lo, color="purple", lw=1.0, ls="--", alpha=0.7,
                    label=f"Conformal 90% (q={q_conformal:.2f})")
            ax.plot(t_fc, conf_hi, color="purple", lw=1.0, ls="--", alpha=0.7)
        ax.axvline(INPUT_LEN, color="gray", ls=":", lw=1)

        title = (
            f"Sample {i+1}  |  Ale σ: {ale_std[i].mean():.4f}  "
            f"Epi σ: {epi_std[i].mean():.4f}"
        )
        if r_prob is not None:
            title += (
                f"  |  Risk — True: {'YES' if r_true[i] > 0.5 else 'NO'}  "
                f"Pred: {r_prob[i]:.2f}"
            )
        ax.set_title(title)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Normalised ECG")
        ax.legend(loc="upper right", fontsize=7, ncol=4)

    plt.suptitle("Probabilistic ECG Forecasting — Ensemble + MC Dropout Uncertainty",
                 fontsize=13, y=1.005)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "predictions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[test] predictions.png              → {out}")


# ---------------------------------------------------------------------------
# Plot 2 — Uncertainty horizon
# ---------------------------------------------------------------------------

def plot_uncertainty_horizon(
    models:     list[ProbabilisticLSTM],
    loader,
    device:     torch.device,
    K:          int,
    max_batches: int = 40,
) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_ale, all_epi = [], []

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x_sig        = batch["x_signal"].to(device)
        x_ann        = batch["x_annot"].to(device)
        x_feat       = batch["x_feat"].to(device)
        x_hrv        = batch["x_hrv"].to(device)
        x_feat_mask  = batch["x_feat_mask"].to(device)
        x_beat_event = batch["x_beat_event"].to(device)
        _, _, epi_var, ale_var, _ = ensemble_mc_predict(
            models, x_sig, x_ann, x_feat, K=K, x_hrv=x_hrv, 
            x_feat_mask=x_feat_mask, x_beat_event=x_beat_event, n_mc=20
        )
        all_ale.append(ale_var.cpu().numpy())
        all_epi.append(epi_var.cpu().numpy())

    ale = np.sqrt(np.concatenate(all_ale).mean(axis=0))
    epi = np.sqrt(np.concatenate(all_epi).mean(axis=0))
    t   = np.arange(FORECAST_LEN) / 360

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, ale, color="orange", label="Aleatoric σ (data noise)")
    ax.plot(t, epi, color="purple", label="Epistemic σ (model uncertainty)")
    ax.fill_between(t, 0, ale, alpha=0.15, color="orange")
    ax.fill_between(t, 0, epi, alpha=0.15, color="purple")
    ax.set_xlabel("Forecast horizon (seconds)")
    ax.set_ylabel("Standard Deviation (normalised)")
    ax.set_title("Uncertainty Decomposition Across Forecast Horizon")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "uncertainty_horizon.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[test] uncertainty_horizon.png      → {out}")


# ---------------------------------------------------------------------------
# Plot 3 — Risk analysis
# ---------------------------------------------------------------------------

def plot_risk_analysis(
    models:      list[ProbabilisticLSTM],
    loader,
    device:      torch.device,
    K:           int,
    opt_thresh:  float = 0.5,
    max_batches: int = 60,
) -> None:
    """Plot risk score distribution and ROC curve.

    opt_thresh : Youden's-J optimal threshold computed by evaluate();
                 shown alongside the 0.5 baseline on the distribution plot.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_probs, all_labels = [], []

    for model in models:
        model.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x_sig        = batch["x_signal"].to(device)
            x_ann        = batch["x_annot"].to(device)
            x_feat       = batch["x_feat"].to(device)
            x_hrv        = batch["x_hrv"].to(device)
            y_risk       = batch["y_risk"]
            x_feat_mask  = batch["x_feat_mask"].to(device)
            x_beat_event = batch["x_beat_event"].to(device)

            batch_risks = []
            for model in models:
                _, risk_logit = model(x_sig, x_ann, x_feat, x_hrv=x_hrv,
                                    x_feat_mask=x_feat_mask, x_beat_event=x_beat_event)
                if risk_logit is None:
                    print("[test] No risk head — skipping risk_analysis.png")
                    return
                risk_logit_cal = risk_logit / getattr(model, 'temperature', 1.0)
                batch_risks.append(torch.sigmoid(risk_logit_cal).squeeze(-1))

            prob = torch.stack(batch_risks).mean(dim=0)
            all_probs.append(prob.cpu().numpy())
            all_labels.append(y_risk.numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    # ROC curve
    desc = np.argsort(-probs)
    ys   = labels[desc]
    nP, nN = ys.sum(), (1 - ys).sum()
    tpr = np.concatenate([[0], np.cumsum(ys)      / max(nP, 1), [1]])
    fpr = np.concatenate([[0], np.cumsum(1 - ys)  / max(nN, 1), [1]])
    auc = float(np.trapezoid(tpr, fpr))

    # Youden's J on ROC curve to mark optimal operating point
    j_idx   = np.argmax(tpr - fpr)
    opt_fpr = float(fpr[j_idx])
    opt_tpr = float(tpr[j_idx])

    # F1 at both thresholds for legend
    _, _, f1_opt  = f1_at_threshold(probs, labels, opt_thresh)
    _, _, f1_half = f1_at_threshold(probs, labels, 0.5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(0, 1, 30)
    ax1.hist(probs[labels == 0], bins=bins, alpha=0.6, color="steelblue", density=True,
             label=f"No arrhythmia  (n={int((labels==0).sum()):,})")
    ax1.hist(probs[labels == 1], bins=bins, alpha=0.6, color="crimson",   density=True,
             label=f"Arrhythmia  (n={int((labels==1).sum()):,})")
    ax1.axvline(0.5,       color="black",  ls="--", lw=1.2, label=f"0.5 baseline  (F1={f1_half:.3f})")
    ax1.axvline(opt_thresh, color="green", ls="--", lw=1.5, label=f"Youden optimal t={opt_thresh:.2f}  (F1={f1_opt:.3f})")
    ax1.set_xlabel("Predicted Risk Probability")
    ax1.set_ylabel("Density")
    ax1.set_title("Risk Score Distribution by Class")
    ax1.legend(fontsize=8)

    ax2.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.3f}")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax2.scatter([opt_fpr], [opt_tpr], color="green", zorder=5, s=80,
                label=f"Youden J point  t={opt_thresh:.2f}")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve — Arrhythmia Risk")
    ax2.legend(fontsize=9)

    plt.suptitle("Arrhythmia Risk Head Evaluation — Ensemble + Youden's J Threshold", fontsize=13)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "risk_analysis.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[test] risk_analysis.png            → {out}")


# ---------------------------------------------------------------------------
# Plot 4 — Calibration reliability diagram
# ---------------------------------------------------------------------------

def plot_calibration(
    models:      list[ProbabilisticLSTM],
    loader,
    device:      torch.device,
    K:           int,
    q_conformal: float | None = None,
    max_batches: int = 60,
) -> float:
    """Plot expected vs actual Gaussian coverage and compute ECE.

    If q_conformal is given, also marks the conformal coverage level on the plot.

    Returns ECE (Expected Calibration Error) for the signal head.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_mu, all_sigma, all_y = [], [], []

    for model in models:
        model.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x_sig        = batch["x_signal"].to(device)
            x_ann        = batch["x_annot"].to(device)
            x_feat       = batch["x_feat"].to(device)
            y_sig        = batch["y_signal"]
            x_hrv        = batch["x_hrv"].to(device)
            x_feat_mask  = batch["x_feat_mask"].to(device)
            x_beat_event = batch["x_beat_event"].to(device)

            batch_means = []
            batch_vars  = []
            for model in models:
                sig_out, _ = model(x_sig, x_ann, x_feat, x_hrv=x_hrv,
                                   x_feat_mask=x_feat_mask, x_beat_event=x_beat_event)
                if sig_out.shape[-1] == 1:
                    print("[test] Deterministic mode — skipping calibration.png")
                    return float("nan")
                batch_means.append(signal_mean(sig_out, K=K))
                batch_vars.append(signal_variance(sig_out, K=K))

            mu    = torch.stack(batch_means).mean(dim=0)
            sigma = torch.sqrt(torch.stack(batch_vars).mean(dim=0)).clamp(min=1e-6)
            all_mu.append(mu.cpu().numpy())
            all_sigma.append(sigma.cpu().numpy())
            all_y.append(y_sig.numpy())

    mu_arr    = np.concatenate(all_mu).ravel()
    sigma_arr = np.concatenate(all_sigma).ravel()
    y_arr     = np.concatenate(all_y).ravel()

    conf_levels = np.linspace(0.05, 0.95, 19)
    actual_cov  = []

    std_norm = torch.distributions.Normal(0.0, 1.0)
    for conf in conf_levels:
        z   = std_norm.icdf(torch.tensor((1.0 + conf) / 2.0)).item()
        cov = ((y_arr >= mu_arr - z * sigma_arr) & (y_arr <= mu_arr + z * sigma_arr)).mean()
        actual_cov.append(float(cov))

    actual_cov = np.array(actual_cov)
    ece        = float(np.abs(actual_cov - conf_levels).mean())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(conf_levels, actual_cov, "o-", color="steelblue", lw=2,
            label=f"Ensemble  (ECE = {ece:.4f})")
    ax.fill_between(conf_levels, conf_levels, actual_cov, alpha=0.15, color="steelblue")

    # Mark conformal coverage level
    if q_conformal is not None:
        # Compute actual coverage at the conformal interval
        conf_cov = float(
            ((y_arr >= mu_arr - q_conformal * sigma_arr) &
             (y_arr <= mu_arr + q_conformal * sigma_arr)).mean()
        )
        ax.axhline(conf_cov, color="purple", lw=1.2, ls=":",
                   label=f"Conformal coverage ({conf_cov:.3f})")
        ax.axhline(0.90, color="orange", lw=1.0, ls="--",
                   label="Nominal 90% target")

    ax.set_xlabel("Expected Coverage")
    ax.set_ylabel("Actual Coverage")
    ax.set_title("Calibration Reliability Diagram (Signal Head — Ensemble)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "calibration.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[test] calibration.png              → {out}  ECE={ece:.4f}")
    return ece


# ---------------------------------------------------------------------------
# Plot 5 — Uncertainty conditioned on arrhythmia presence
# ---------------------------------------------------------------------------

def plot_arrhythmia_uncertainty(
    models:      list[ProbabilisticLSTM],
    loader,
    device:      torch.device,
    K:           int,
    max_batches: int = 50,
) -> None:
    """Compare aleatoric and epistemic σ between arrhythmia and normal windows."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ale_arr, ale_norm, epi_arr, epi_norm = [], [], [], []

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x_sig        = batch["x_signal"].to(device)
        x_ann        = batch["x_annot"].to(device)
        x_feat       = batch["x_feat"].to(device)
        x_hrv        = batch["x_hrv"].to(device)
        y_risk       = batch["y_risk"]
        x_feat_mask  = batch["x_feat_mask"].to(device)
        x_beat_event = batch["x_beat_event"].to(device)

        _, _, epi_var, ale_var, _ = ensemble_mc_predict(
            models, x_sig, x_ann, x_feat, K=K, x_hrv=x_hrv, 
            x_feat_mask=x_feat_mask, x_beat_event=x_beat_event, n_mc=20
        )
        ale_np = ale_var.cpu().numpy()
        epi_np = epi_var.cpu().numpy()
        mask   = y_risk.bool().numpy()

        if mask.any():
            ale_arr.append(ale_np[mask])
            epi_arr.append(epi_np[mask])
        if (~mask).any():
            ale_norm.append(ale_np[~mask])
            epi_norm.append(epi_np[~mask])

    if not ale_arr or not ale_norm:
        print("[test] Not enough arrhythmia samples — skipping arrhythmia_uncertainty.png")
        return

    t = np.arange(FORECAST_LEN) / 360

    ale_a = np.sqrt(np.concatenate(ale_arr).mean(axis=0))
    ale_n = np.sqrt(np.concatenate(ale_norm).mean(axis=0))
    epi_a = np.sqrt(np.concatenate(epi_arr).mean(axis=0))
    epi_n = np.sqrt(np.concatenate(epi_norm).mean(axis=0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(t, ale_a, color="crimson",   lw=2, label="Arrhythmia windows")
    ax1.plot(t, ale_n, color="steelblue", lw=2, label="Normal windows")
    ax1.fill_between(t, ale_n, ale_a, alpha=0.15, color="crimson")
    ax1.set_title("Aleatoric Uncertainty by Arrhythmia Status")
    ax1.set_xlabel("Forecast horizon (s)")
    ax1.set_ylabel("Aleatoric σ")
    ax1.legend()

    ax2.plot(t, epi_a, color="crimson",   lw=2, label="Arrhythmia windows")
    ax2.plot(t, epi_n, color="steelblue", lw=2, label="Normal windows")
    ax2.fill_between(t, epi_n, epi_a, alpha=0.15, color="crimson")
    ax2.set_title("Epistemic Uncertainty by Arrhythmia Status")
    ax2.set_xlabel("Forecast horizon (s)")
    ax2.set_ylabel("Epistemic σ")
    ax2.legend()

    plt.suptitle("Uncertainty Conditioned on Arrhythmia Presence — Ensemble", fontsize=13)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "arrhythmia_uncertainty.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[test] arrhythmia_uncertainty.png   → {out}")


# ---------------------------------------------------------------------------
# Plot 6 — Temporal attention heatmap (first ensemble member only)
# ---------------------------------------------------------------------------

def plot_attention_weights(
    models:     list[ProbabilisticLSTM],
    loader,
    device:     torch.device,
    K:          int,
    n_examples: int = 4,
) -> None:
    """Visualise which input timesteps the first ensemble member attends to."""
    model = models[0]
    if not getattr(model, "use_attention", False):
        print("[test] No attention mechanism — skipping attention_weights.png")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    batch        = next(iter(loader))
    x_sig        = batch["x_signal"]
    x_ann        = batch["x_annot"]
    x_feat       = batch["x_feat"]
    y_sig        = batch["y_signal"]
    y_risk       = batch["y_risk"]
    x_hrv        = batch["x_hrv"]
    x_feat_mask  = batch["x_feat_mask"]
    x_beat_event = batch["x_beat_event"]
    x_sig_d  = x_sig.to(device)
    x_ann_d  = x_ann.to(device)
    x_feat_d = x_feat.to(device)
    x_hrv_d  = x_hrv.to(device)

    model.eval()
    with torch.no_grad():
        sig_out, risk_logit, attn_weights = model(
            x_sig_d, x_ann_d, x_feat_d, x_hrv=x_hrv_d, 
            x_feat_mask=x_feat_mask.to(device),
            x_beat_event=x_beat_event.to(device), return_attn=True
        )

    if attn_weights is None:
        return

    attn   = attn_weights.cpu().numpy()         # (batch, input_len)
    x_np   = x_sig[:, :, 0].numpy()             # (batch, input_len) — lead 0
    ann_np = x_ann.numpy()                      # (batch, input_len)
    r_prob = (torch.sigmoid(risk_logit / getattr(model, 'temperature', 1.0)).squeeze(-1).cpu().numpy()
              if risk_logit is not None else None)
    r_true = y_risk.numpy()

    n = min(n_examples, len(x_np))
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n))
    if n == 1:
        axes = [axes]

    t = np.arange(INPUT_LEN)

    for i, ax in enumerate(axes):
        ax.plot(t, x_np[i], color="steelblue", lw=0.8, alpha=0.8, label="ECG signal")

        sig_range   = x_np[i].max() - x_np[i].min() + 1e-8
        attn_scaled = attn[i] / (attn[i].max() + 1e-8) * sig_range * 0.5
        ax.fill_between(
            t, x_np[i].min(), x_np[i].min() + attn_scaled,
            alpha=0.45, color="orange", label="Attention weight",
        )

        ab_pos = np.where(ann_np[i] == 2)[0]
        if len(ab_pos):
            ax.scatter(ab_pos, x_np[i][ab_pos], color="crimson", s=35,
                       zorder=5, label="Abnormal beat")

        nm_pos = np.where(ann_np[i] == 1)[0][::3]
        if len(nm_pos):
            ax.scatter(nm_pos, x_np[i][nm_pos], color="green", s=18,
                       zorder=4, marker="^", alpha=0.7, label="Normal beat")

        title = f"Sample {i+1}  |  Peak attention @ t={attn[i].argmax()}"
        if r_prob is not None:
            title += (
                f"  |  Risk — True: {'YES' if r_true[i] > 0.5 else 'NO'}  "
                f"Pred: {r_prob[i]:.2f}"
            )
        ax.set_title(title)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Normalised ECG")
        ax.legend(loc="upper right", fontsize=8, ncol=4)

    plt.suptitle("Temporal Attention Weights Over Input Window (model_0)", fontsize=13, y=1.005)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "attention_weights.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[test] attention_weights.png        → {out}")

# ---------------------------------------------------------------------------
# Plot 7 — Uncertainty analysis plot
# ---------------------------------------------------------------------------

def plot_uncertainty_scatter_decomposed(errors, total, ale, epi, save_dir="results"):

    plt.figure(figsize=(6,5))

    idx = np.random.choice(len(errors), size=min(5000, len(errors)), replace=False)

    plt.scatter(total[idx], errors[idx], alpha=0.3, s=5, label="Total", color="black")
    plt.scatter(ale[idx],   errors[idx], alpha=0.3, s=5, label="Aleatoric", color="orange")
    plt.scatter(epi[idx],   errors[idx], alpha=0.3, s=5, label="Epistemic", color="purple")

    plt.xlabel("Uncertainty")
    plt.ylabel("Prediction Error")
    plt.title("Error vs Uncertainty (Decomposed)")

    plt.legend()
    plt.grid(alpha=0.3)

    out = os.path.join(save_dir, "uncertainty_scatter_decomposed.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[test] uncertainty_scatter_decomposed.png → {out}")

# ---------------------------------------------------------------------------
# Plot 8 — Uncertainty quantile plot
# ---------------------------------------------------------------------------

def plot_quantile_decomposed(errors, total, ale, epi, save_dir="results", n_bins=10):

    def compute_curve(errors, uncs):
        idx = np.argsort(uncs)
        errors_sorted = errors[idx]
        bins = np.array_split(np.arange(len(errors)), n_bins)
        return [errors_sorted[b].mean() for b in bins]

    total_curve = compute_curve(errors, total)
    ale_curve   = compute_curve(errors, ale)
    epi_curve   = compute_curve(errors, epi)

    plt.figure(figsize=(6,5))

    x = range(1, n_bins+1)
    plt.plot(x, total_curve, marker='o', label="Total", color="black")
    plt.plot(x, ale_curve,   marker='o', label="Aleatoric", color="orange")
    plt.plot(x, epi_curve,   marker='o', label="Epistemic", color="purple")

    plt.xlabel("Uncertainty Quantile (low → high)")
    plt.ylabel("Mean Prediction Error")
    plt.title("Error vs Uncertainty Quantile (Decomposed)")

    plt.legend()
    plt.grid(alpha=0.3)

    out = os.path.join(save_dir, "quantile_decomposed.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[test] quantile_decomposed.png → {out}")

# ---------------------------------------------------------------------------
# Plot 9 — Uncertainty by retention plot
# ---------------------------------------------------------------------------

def plot_retention_decomposed(errors, total, ale, epi, save_dir="results"):

    def compute_curve(errors, uncs):
        idx = np.argsort(uncs)
        errors_sorted = errors[idx]

        fractions = np.linspace(0.1, 1.0, 10)
        N = len(errors_sorted)

        return [
            errors_sorted[:int(f * N)].mean()
            for f in fractions
        ], fractions

    total_curve, f = compute_curve(errors, total)
    ale_curve, _   = compute_curve(errors, ale)
    epi_curve, _   = compute_curve(errors, epi)

    plt.figure(figsize=(6,5))

    plt.plot(f, total_curve, marker='o', label="Total", color="black")
    plt.plot(f, ale_curve,   marker='o', label="Aleatoric", color="orange")
    plt.plot(f, epi_curve,   marker='o', label="Epistemic", color="purple")

    plt.xlabel("Fraction of Retained Samples")
    plt.ylabel("Mean Prediction Error")
    plt.title("Retention Curve (Uncertainty Decomposition)")

    plt.legend()
    plt.grid(alpha=0.3)

    out = os.path.join(save_dir, "retention_decomposed.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[test] retention_decomposed.png → {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K      = K_MDN if USE_MDN else 1
    print(f"[test] Device: {device}  K={K}\n")

    # 1. Load ensemble
    models = load_ensemble(MODEL_DIR, device)

    # 2. Get data loaders (4-tuple: train, val, train_ds, val_ds
    _, val_loader, train_ds, val_ds= get_augmented_dataloaders(
        data_folder=DATA_FOLDER, input_len=INPUT_LEN, forecast_len=FORECAST_LEN,
        stride=STRIDE, batch_size=BATCH_SIZE, split=TRAIN_VAL_SPLIT, seed=SEED,
    )
    cal_loader = val_loader

    # 3. Conformal calibration on cal_loader (signal)
    q_conformal = conformal_calibrate(models, cal_loader, device, K=K, alpha=0.10)

    # 3b. Conformal calibration for risk head
    q_risk = conformal_calibrate_risk(models, cal_loader, device, alpha=CONFORMAL_ALPHA_RISK)

    # 4. Quantitative evaluation on val_loader
    metrics = evaluate(models, val_loader, device, K=K, q_conformal=q_conformal)
    errors = metrics["all_errors"]
    total  = metrics["all_total_unc"]
    ale    = metrics["all_ale_unc"]
    epi    = metrics["all_epi_unc"]
    opt_t   = metrics["opt_thresh"]
    print(f"\n[test] ── Evaluation Results ───────────────────────────────────")
    print(f"[test]   Signal MSE              : {metrics['mse']:.6f}")
    print(f"[test]   Signal NLL (β=0)        : {metrics['nll']:.6f}")
    print(f"[test]   Signal CRPS             : {metrics['crps']:.6f}")
    print(f"[test]   Winkler Score (90%)     : {metrics['winkler']:.6f}")
    print(f"[test] ── Arrhythmia Classification ─────────────────────────────")
    print(f"[test]   AUC-ROC                 : {metrics['risk_auc']:.4f}")
    print(f"[test]   Threshold = 0.50 ·  Acc={metrics['risk_acc']:.4f}  F1={metrics['risk_f1_half']:.4f}")
    print(f"[test]   Youden J  t={opt_t:.3f}·  Acc={metrics['risk_acc_opt']:.4f}  "
          f"Prec={metrics['risk_prec']:.4f}  Recall={metrics['risk_recall']:.4f}  F1={metrics['risk_f1']:.4f}")
    print(f"[test] ─────────────────────────────────────────────────────────\n")

    # 4b. Conformal risk set size distribution
    if not math.isnan(q_risk):
        print(f"[test] Conformal risk threshold q_risk={q_risk:.4f}")
        print(f"[test]   Include class 0 (no arrhythmia) if prob <= {q_risk:.4f}")
        print(f"[test]   Include class 1 (arrhythmia)    if prob >= {1.0 - q_risk:.4f}")

    # 5. Per-beat-type metrics
    per_beat_type_evaluate(models, val_loader, device, K=K)

    # 6. All plots
    plot_predictions(models, val_loader, device, K=K, q_conformal=q_conformal)
    plot_uncertainty_horizon(models, val_loader, device, K=K)
    plot_risk_analysis(models, val_loader, device, K=K, opt_thresh=opt_t)
    ece = plot_calibration(models, val_loader, device, K=K, q_conformal=q_conformal)
    plot_arrhythmia_uncertainty(models, val_loader, device, K=K)
    plot_attention_weights(models, val_loader, device, K=K)
    plot_uncertainty_scatter_decomposed(errors, total, ale, epi)
    plot_quantile_decomposed(errors, total, ale, epi)
    plot_retention_decomposed(errors, total, ale, epi)

    # 7. Diebold-Mariano test: full ensemble vs first member alone
    if len(models) > 1:
        errors_ens, errors_m0 = collect_forecast_errors(models, [models[0]], val_loader, device, K)
        dm_stat, dm_p = diebold_mariano_test(errors_ens, errors_m0)
        print(f"[test]   DM test (ensemble vs model_0): stat={dm_stat:.3f}  p={dm_p:.4f}")

    # 8. Summary
    print(f"\n[test] ── Summary ───────────────────────────────────────────────")
    print(f"[test]   Ensemble size            : {len(models)}")
    print(f"[test]   K (mixture components)   : {K}")
    print(f"[test]   Conformal q (alpha=0.10) : {q_conformal:.4f}")
    if not math.isnan(q_risk):
        print(f"[test]   Conformal q_risk         : {q_risk:.4f}")
    print(f"[test]   ECE (calibration error)  : {ece:.4f}")
    print(f"[test]   Winkler Score (90%)      : {metrics['winkler']:.6f}")
    print(f"[test]   All figures saved to '{RESULTS_DIR}/'")


if __name__ == "__main__":
    main()

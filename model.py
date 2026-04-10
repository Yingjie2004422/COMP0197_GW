# model.py
# Dual-task probabilistic LSTM for ECG signal forecasting + arrhythmia risk.
#
# Architecture overview
# ---------------------
# Inputs per timestep (concatenated before LSTM):
#   1. Raw normalised ECG voltage (2 leads)                                   — (2,)
#   2. Beat-type embedding (0/1/2 token)                                      — (embed_dim,)
#   3. Beat morphology features + observation mask [opt.] (4 channels)        — (8,)  if USE_RR_FEATURES
#      [RR, dRR, peak_amp, dAmp, RR_mask, dRR_mask, amp_mask, dAmp_mask]
#      Mask channel tells model which values are real beat observations vs forward-filled
#   4. Sparse beat-even stream (if x_beat_event is provided)                  - (input_len, 7)
#      [is_beat, is_normal, is_abnormal, RR, dRR, amp, dAmp] at actual beat positions
#
# Bidirectional LSTM encoder → BiLSTM projection → multi-head temporal attention
# pooling [opt.] → HRV feature fusion [opt.] → shared hidden state
# → Seq2Seq autoregressive decoder → two independent heads:
#   Signal head  → Gaussian (K=1) or Mixture of K Gaussians over future ECG
#                  K=1: outputs (μ, pre_σ); σ = softplus(pre_σ) + ε
#                  K>1: outputs (log_π×K | μ×K | pre_σ×K) stacked
#   Risk head    → binary logit for "abnormal beat in forecast window"
#
# Improvements over baseline
# --------------------------
#   Bidirectional LSTM     : processes input in both directions; projection layer
#                            maps 2*hidden_size back to hidden_size.
#   MultiHeadTemporalAttn  : multi-head attention over all LSTM timesteps.
#   Seq2Seq decoder        : autoregressive 1-layer LSTM decoder with teacher forcing.
#   Dual-lead input        : uses both ECG leads (INPUT_CHANNELS=2).
#   HRV feature fusion     : SDNN, RMSSD, pNN50 fused after attention pooling.
#   Beta-NLL loss          : σ^(2β) weighting (Seitzer et al., 2022).
#   gaussian_crps          : closed-form CRPS — proper scoring rule.
#   MDN (K>1)              : Mixture Density Network, K Gaussian components.
#   Focal loss             : down-weights easy negatives for risk head.
#   CRPS training loss     : differentiable CRPS used as signal objective.
#   4-channel features     : RR, dRR, peak_amp, dAmp.
#   Label smoothing        : soft targets for risk head BCE loss.
#
# Ablation flags (sourced from config.py, overridable at construction):
#   use_attention   : toggle temporal attention pooling
#   use_rr_features : include beat morphology channels in LSTM input
#   use_layer_norm  : apply LayerNorm to the pooled hidden state
#   use_risk_head   : add the arrhythmia risk classification head
#   deterministic   : output mean only, MSE loss (baseline model)
#   K               : number of Gaussian mixture components (1 = single Gaussian)
#
# GenAI assistance: used to draft the dual-head structure, attention mechanism,
# Beta-NLL formulation, and MDN extension; verified against Seitzer et al. (2022),
# Bishop (1994) MDN paper, and Kendall & Gal (2017). Extended with bidirectional
# LSTM, multi-head attention, seq2seq decoder, dual-lead input, HRV fusion,
# label smoothing.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal as _TorchNormal

from config import (
    HIDDEN_SIZE, NUM_LAYERS, DROPOUT, FORECAST_LEN,
    EMBED_DIM, NUM_BEAT_CLASSES, RISK_LAMBDA,
    USE_LAYER_NORM, USE_RR_FEATURES, USE_RISK_HEAD, DETERMINISTIC,
    USE_ATTENTION, BETA_NLL, N_FEAT_CHANNELS,
    USE_MDN, K_MDN, USE_FOCAL_LOSS, FOCAL_GAMMA, USE_CRPS_LOSS,
    INPUT_CHANNELS, N_ATTN_HEADS, N_HRV_FEATURES,
    TEACHER_FORCING_RATIO, LABEL_SMOOTHING,
)


# ---------------------------------------------------------------------------
# Module-level helpers for single Gaussian and MDN output
# ---------------------------------------------------------------------------

def signal_mean(output: torch.Tensor, K: int = 1) -> torch.Tensor:
    """Extract the predictive mean from a signal head output tensor.

    Parameters
    ----------
    output : (batch, forecast_len, 2) for K=1
             (batch, forecast_len, 3*K) for K>1  [logpi×K | mu×K | pre_sigma×K]
    K      : number of mixture components

    Returns
    -------
    mean : (batch, forecast_len)
    """
    if K == 1:
        return output[..., 0]
    # MDN: softmax weights × component means
    log_pi = output[..., :K]                         # (batch, T, K)
    mu     = output[..., K:2*K]                      # (batch, T, K)
    pi     = torch.softmax(log_pi, dim=-1)
    return (pi * mu).sum(dim=-1)                     # (batch, T)


def signal_variance(output: torch.Tensor, K: int = 1) -> torch.Tensor:
    """Extract the total predictive variance from a signal head output tensor.

    For K>1: mixture variance = E[sigma^2] + Var[mu_k] (law of total variance).

    Parameters
    ----------
    output : (batch, forecast_len, 2) for K=1
             (batch, forecast_len, 3*K) for K>1
    K      : number of mixture components

    Returns
    -------
    variance : (batch, forecast_len)  (always >= 0)
    """
    if K == 1:
        return (F.softplus(output[..., 1]) + 1e-6) ** 2

    log_pi    = output[..., :K]                      # (batch, T, K)
    mu        = output[..., K:2*K]                   # (batch, T, K)
    pre_sigma = output[..., 2*K:]                    # (batch, T, K)
    sigma     = F.softplus(pre_sigma) + 1e-6         # (batch, T, K)
    pi        = torch.softmax(log_pi, dim=-1)

    mu_mix    = (pi * mu).sum(dim=-1, keepdim=True)  # (batch, T, 1)
    e_var     = (pi * sigma ** 2).sum(dim=-1)                # E[sigma^2]
    var_mu    = (pi * (mu - mu_mix) ** 2).sum(dim=-1)        # Var[mu_k]
    return e_var + var_mu                            # (batch, T)


# ---------------------------------------------------------------------------
# Temporal attention modules
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """Additive (Bahdanau-style) attention over LSTM output timesteps.

    Kept for backward compatibility; MultiHeadTemporalAttention is used
    when use_attention=True.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self, lstm_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        lstm_out : (batch, seq_len, hidden_size)

        Returns
        -------
        context : (batch, hidden_size)  — weighted sum of hidden states
        weights : (batch, seq_len)      — normalised attention weights
        """
        scores  = self.score(lstm_out).squeeze(-1)   # (batch, seq_len)
        weights = torch.softmax(scores, dim=-1)       # (batch, seq_len)
        context = (weights.unsqueeze(-1) * lstm_out).sum(dim=1)  # (batch, hidden)
        return context, weights


class MultiHeadTemporalAttention(nn.Module):
    """Multi-head attention over LSTM output timesteps.

    Uses nn.MultiheadAttention with a mean-pooled query to produce a
    context vector for each batch element.
    """

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

    def forward(
        self, lstm_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        lstm_out : (batch, seq_len, hidden_size)

        Returns
        -------
        context : (batch, hidden_size)
        weights : (batch, seq_len)
        """
        # lstm_out: (batch, seq_len, hidden_size)
        query = lstm_out.mean(dim=1, keepdim=True)  # (batch, 1, hidden_size)
        context, weights = self.attn(query, lstm_out, lstm_out, need_weights=True, average_attn_weights=True)
        context = context.squeeze(1)       # (batch, hidden_size)
        weights = weights.squeeze(1)       # (batch, seq_len)
        return context, weights


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ProbabilisticLSTM(nn.Module):
    """Annotation-aware probabilistic LSTM for ECG forecasting + arrhythmia risk.

    Parameters
    ----------
    input_size       : raw signal channels (2 for dual-lead ECG)
    hidden_size      : LSTM hidden dimension
    num_layers       : number of stacked LSTM layers
    forecast_len     : number of future timesteps to predict
    dropout          : dropout probability (inter-layer LSTM + pre-head)
    embed_dim        : beat-type embedding dimension
    num_beat_classes : vocabulary size for beat tokens (0/1/2 → 3)
    use_attention    : use multi-head temporal attention pooling over LSTM outputs
    use_rr_features  : include beat morphology features in LSTM input (4 ch)
    use_layer_norm   : apply LayerNorm to the pooled hidden state
    use_risk_head    : include the arrhythmia risk head
    deterministic    : output mean only, MSE baseline mode
    K                : number of Gaussian mixture components (1 = single Gaussian)
    """

    def __init__(
        self,
        input_size:       int   = INPUT_CHANNELS,
        hidden_size:      int   = HIDDEN_SIZE,
        num_layers:       int   = NUM_LAYERS,
        forecast_len:     int   = FORECAST_LEN,
        dropout:          float = DROPOUT,
        embed_dim:        int   = EMBED_DIM,
        num_beat_classes: int   = NUM_BEAT_CLASSES,
        use_attention:    bool  = False,       # default False for checkpoint compat
        use_rr_features:  bool  = USE_RR_FEATURES,
        use_layer_norm:   bool  = USE_LAYER_NORM,
        use_risk_head:    bool  = USE_RISK_HEAD,
        deterministic:    bool  = DETERMINISTIC,
        K:                int   = 1,
        num_heads:       int   =  N_ATTN_HEADS
    ) -> None:
        super().__init__()

        self.forecast_len           = forecast_len
        self.use_attention          = use_attention
        self.use_rr_features        = use_rr_features
        self.use_risk_head          = use_risk_head
        self.deterministic          = deterministic
        self.K                      = K
        self.num_heads              = num_heads
        # Mutable so train.py can apply the scheduled annealing schedule
        self.teacher_forcing_ratio  = TEACHER_FORCING_RATIO

        # Beat-type embedding.
        # padding_idx=0 keeps the "no beat" token as the zero vector.
        self.beat_embedding = nn.Embedding(
            num_beat_classes, embed_dim, padding_idx=0
        )

        # Beat even encoder
        # x_beat_event is a sparse tensore with non-zero values at actual beat positions
        # beat_event_proj fuses resulting context vector with LSTM hidden state
        self.beat_event_encoder = nn.Sequential(
            nn.Conv1d(in_channels=7, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=hidden_size, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.beat_event_proj = nn.Linear(hidden_size * 2, hidden_size)

        # LSTM input size: signal + embedding + optional feature channels
        feat_channels  = N_FEAT_CHANNELS if use_rr_features else 0
        lstm_input_dim = INPUT_CHANNELS + embed_dim + feat_channels

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size  = lstm_input_dim,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
        )

        # Projection from bidirectional (2*hidden_size) back to hidden_size
        self.bilstm_proj = nn.Linear(2 * hidden_size, hidden_size)

        # Optional temporal attention — multi-head replaces naive last-timestep pooling.
        if use_attention:
            self.attention = MultiHeadTemporalAttention(hidden_size, num_heads=N_ATTN_HEADS)

        # LayerNorm on the pooled hidden state.
        self.layer_norm = nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()

        # Dropout on the hidden state — also exploited by MC Dropout at test time.
        self.dropout = nn.Dropout(dropout)

        # HRV feature fusion: project (hidden_size + N_HRV_FEATURES) -> hidden_size
        self.hrv_proj = nn.Linear(hidden_size + N_HRV_FEATURES, hidden_size)

        # Signal head / decoder
        if not deterministic:
            # Seq2Seq decoder
            self.decoder_lstm = nn.LSTM(
                input_size=1,  # prev lead-0 signal value
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.fc_dec_h = nn.Linear(hidden_size, hidden_size)  # encoder context -> decoder init hidden
            self.fc_dec_c = nn.Linear(hidden_size, hidden_size)  # encoder context -> decoder init cell
            if K > 1:
                self.decoder_head = nn.Linear(hidden_size, 3 * K)  # MDN per-step
            else:
                self.decoder_head = nn.Linear(hidden_size, 2)      # (mu, pre_sigma) per-step
        else:
            self.fc_mean = nn.Linear(hidden_size, forecast_len)

        # Risk head — omitted in deterministic / baseline mode
        if use_risk_head and not deterministic:
            self.fc_risk = nn.Linear(hidden_size, 1)

    # ------------------------------------------------------------------
    def forward(
        self,
        x_signal:    torch.Tensor,          # (batch, input_len, 2)     float32
        x_annot:     torch.Tensor,          # (batch, input_len)         int64
        x_feat:      torch.Tensor | None,   # (batch, input_len, 4)     float32
        x_hrv:       torch.Tensor | None = None,   # (batch, 3)          float32
        x_feat_mask: torch.Tensor | None = None,   # (batch, input_len, 4) float32 - 1 where observed, 0 where forward-filled
        x_beat_event:torch.Tensor | None = None,   # (batch, input_len, 7) float32 - sparse beat-event stream
        y_signal:    torch.Tensor | None = None,   # (batch, forecast_len) float32, for teacher forcing
        return_attn: bool = False,
    ) -> tuple:
        """
        Returns
        -------
        signal_out : (batch, forecast_len, 3*K) — MDN: [logpi×K | mu×K | pre_sigma×K]
                  or (batch, forecast_len, 2)   — single Gaussian [μ, pre_σ]
                  or (batch, forecast_len, 1)   — deterministic [μ]
        risk_logit : (batch, 1)  or  None
        attn_weights (only when return_attn=True) : (batch, input_len) or None
        """
        emb   = self.beat_embedding(x_annot)           # (batch, input_len, embed_dim)
        parts = [x_signal, emb]
        if self.use_rr_features and x_feat is not None:
            if x_feat_mask is not None:
                parts.append(torch.cat([x_feat, x_feat_mask], dim=-1))
            else:
                parts.append(x_feat)

        lstm_in     = torch.cat(parts, dim=-1)
        lstm_out, _ = self.lstm(lstm_in)               # (batch, input_len, 2*hidden)

        # Project bidirectional output back to hidden_size
        lstm_out_proj = self.bilstm_proj(lstm_out)     # (batch, input_len, hidden)

        # Pooling: attention over all timesteps (preferred) or last state only
        if self.use_attention:
            h, attn_weights = self.attention(lstm_out_proj)
        else:
            h            = lstm_out_proj[:, -1, :]
            attn_weights = None

        h = self.layer_norm(h)
        h = self.dropout(h)

        # HRV feature fusion
        if x_hrv is not None and hasattr(self, 'hrv_proj'):
            h = torch.relu(self.hrv_proj(torch.cat([h, x_hrv], dim=-1)))

        # Beat event encoder and fusing with h 
        if x_beat_event is not None:
            be = x_beat_event.permute(0, 2, 1)           # (batch, 7, input_len) for Conv1d
            be_encoded = self.beat_event_encoder(be)       # (batch, hidden_size, input_len)
            be_pooled = be_encoded.mean(dim=-1)            # (batch, hidden_size)
            h = torch.relu(self.beat_event_proj(torch.cat([h, be_pooled], dim=-1)))

        if self.deterministic:
            mu  = self.fc_mean(h)                      # (batch, forecast_len)
            out = (mu.unsqueeze(-1), None)
            return (*out, attn_weights) if return_attn else out

        # Seq2Seq autoregressive decoder
        # h: (batch, hidden_size) — pooled encoder context
        h_dec_0 = torch.tanh(self.fc_dec_h(h)).unsqueeze(0)   # (1, batch, hidden_size)
        c_dec_0 = torch.tanh(self.fc_dec_c(h)).unsqueeze(0)
        dec_hidden = (h_dec_0, c_dec_0)

        start_tok = x_signal[:, -1:, 0:1]  # (batch, 1, 1) — last lead-0 value
        dec_inp = start_tok
        outputs = []

        # Use teacher forcing whenever y_signal is provided (training OR validation loss
        # computation).  Free-running only when y_signal is None (actual inference in
        # test.py, which never passes y_signal).  The previous `self.training and ...`
        # caused validation to always run free-running from an untrained decoder,
        # producing catastrophically high val loss and triggering early stopping at epoch 1.
        use_tf = y_signal is not None

        for t in range(self.forecast_len):
            dec_out, dec_hidden = self.decoder_lstm(dec_inp, dec_hidden)   # (batch, 1, hidden)
            step_out = self.decoder_head(dec_out)                           # (batch, 1, 2 or 3K)
            outputs.append(step_out)

            # Next decoder input
            if use_tf and torch.rand(1).item() < self.teacher_forcing_ratio:
                dec_inp = y_signal[:, t:t+1].unsqueeze(-1)  # (batch, 1, 1)
            else:
                if self.K > 1:
                    # use mixture mean as next input
                    s = step_out.squeeze(1)              # (batch, 3K)
                    pi = torch.softmax(s[..., :self.K], dim=-1)
                    mu_k = s[..., self.K:2*self.K]
                    mu_prev = (pi * mu_k).sum(-1, keepdim=True).unsqueeze(1)  # (batch, 1, 1)
                    dec_inp = mu_prev
                else:
                    dec_inp = step_out[..., 0:1]          # (batch, 1, 1) — mu

        signal_out = torch.cat(outputs, dim=1)  # (batch, forecast_len, 2 or 3K)

        risk_logit = self.fc_risk(h) if self.use_risk_head else None

        out = (signal_out, risk_logit)
        return (*out, attn_weights) if return_attn else out


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def gaussian_nll_loss(
    signal_out: torch.Tensor,   # (batch, forecast_len, 2)
    target:     torch.Tensor,   # (batch, forecast_len)
    beta:       float = BETA_NLL,
) -> torch.Tensor:
    """Beta-NLL: Gaussian NLL weighted by σ^(2β) to prevent variance collapse.

    Standard NLL:
        NLL = log(σ) + 0.5 * ((y − μ) / σ)²

    Beta-NLL (Seitzer et al., 2022):
        L_β = stop_grad(σ^(2β)) * NLL

    β=0 → standard NLL.  β=0.5 → recommended default; balances mean accuracy
    with variance calibration by down-weighting high-variance predictions.

    σ is obtained via softplus (always > 0, smooth, no saturation).
    """
    mu        = signal_out[..., 0]
    pre_sigma = signal_out[..., 1]
    sigma     = F.softplus(pre_sigma) + 1e-6

    nll = torch.log(sigma) + 0.5 * ((target - mu) / sigma) ** 2

    if beta > 0.0:
        # Detach so the weight does not propagate gradients back through sigma
        weight = sigma.detach().pow(2.0 * beta)
        nll    = weight * nll

    return nll.mean()


def mdn_nll_loss(
    output: torch.Tensor,   # (batch, forecast_len, 3*K)
    target: torch.Tensor,   # (batch, forecast_len)
    K:      int,
) -> torch.Tensor:
    """Negative log-likelihood for a K-component Mixture Density Network.

    log p(y) = logsumexp_k [ log π_k + log N(y; μ_k, σ_k) ]

    Parameters
    ----------
    output : (batch, T, 3*K)  — [logpi×K | mu×K | pre_sigma×K]
    target : (batch, T)
    K      : number of mixture components
    """
    log_pi    = F.log_softmax(output[..., :K],  dim=-1)   # (B, T, K)
    mu        = output[..., K:2*K]                         # (B, T, K)
    pre_sigma = output[..., 2*K:]                          # (B, T, K)
    sigma     = F.softplus(pre_sigma) + 1e-6               # (B, T, K)

    # log N(y; mu_k, sigma_k) for each component
    y_exp = target.unsqueeze(-1).expand_as(mu)             # (B, T, K)
    log_gauss = (
        -0.5 * ((y_exp - mu) / sigma) ** 2
        - torch.log(sigma)
        - 0.5 * np.log(2.0 * np.pi)
    )                                                       # (B, T, K)

    # log-sum-exp over components: log p(y) = logsumexp_k(log_pi_k + log_gauss_k)
    log_prob = torch.logsumexp(log_pi + log_gauss, dim=-1)  # (B, T)
    return -log_prob.mean()


def differentiable_crps_loss(
    output: torch.Tensor,   # (batch, forecast_len, 2)
    target: torch.Tensor,   # (batch, forecast_len)
) -> torch.Tensor:
    """Differentiable closed-form Gaussian CRPS as a training loss.

    CRPS(N(μ, σ), y) = σ * [2φ(z) + z*(2Φ(z) − 1) − 1/√π]
    where z = (y − μ)/σ, φ = PDF, Φ = CDF of N(0,1).

    Returns the mean CRPS as a differentiable tensor (not .item()).
    Uses torch.distributions.Normal so gradients flow through μ and σ.
    """
    mu    = output[..., 0]
    sigma = F.softplus(output[..., 1]) + 1e-6

    std_norm = _TorchNormal(
        torch.zeros_like(mu), torch.ones_like(sigma)
    )
    z   = (target - mu) / sigma
    phi = std_norm.log_prob(z).exp()      # PDF at z
    Phi = std_norm.cdf(z)                 # CDF at z

    crps = sigma * (2.0 * phi + z * (2.0 * Phi - 1.0) - 1.0 / np.sqrt(np.pi))
    return crps.mean()


def focal_bce_loss(
    logit:      torch.Tensor,   # (batch,)
    target:     torch.Tensor,   # (batch,)
    gamma:      float,
    pos_weight: torch.Tensor,   # scalar tensor
) -> torch.Tensor:
    """Focal binary cross-entropy loss (Lin et al., 2017).

    focal_loss_i = (1 - p_t)^gamma * BCE_i

    where p_t = sigmoid(logit) if target=1, else 1 - sigmoid(logit).
    Weighted by pos_weight for class imbalance.

    Parameters
    ----------
    logit      : (batch,)  raw logits
    target     : (batch,)  binary labels {0, 1}
    gamma      : focusing parameter (2.0 recommended)
    pos_weight : scalar tensor for positive class weight
    """
    # Per-sample BCE (no reduction)
    bce = F.binary_cross_entropy_with_logits(
        logit, target, reduction="none"
    )
    # p_t = exp(-bce) when bce = -log(p_t)
    pt       = torch.exp(-bce)
    focal_w  = (1.0 - pt) ** gamma

    # Apply pos_weight manually for the positive class
    weight = torch.where(target > 0.5, pos_weight, torch.ones_like(target))
    return (focal_w * bce * weight).mean()


def gaussian_crps(
    signal_out: torch.Tensor,   # (batch, forecast_len, 2)
    target:     torch.Tensor,   # (batch, forecast_len)
) -> float:
    """Closed-form CRPS for Gaussian forecasts (evaluation metric, not gradient).

    CRPS(N(μ, σ), y) = σ * [2φ(z) + z*(2Φ(z) − 1) − 1/√π]
    where z = (y − μ)/σ.

    Returns float('nan') for deterministic checkpoints.
    """
    if signal_out.shape[-1] < 2:
        return float("nan")

    mu    = signal_out[..., 0]
    sigma = F.softplus(signal_out[..., 1]) + 1e-6

    std_norm = _TorchNormal(torch.zeros_like(mu), torch.ones_like(mu))
    z        = (target - mu) / sigma
    phi      = std_norm.log_prob(z).exp()
    Phi      = std_norm.cdf(z)

    crps = sigma * (2.0 * phi + z * (2.0 * Phi - 1.0) - 1.0 / np.sqrt(np.pi))
    return crps.mean().item()


def combined_loss(
    signal_out:  torch.Tensor,
    risk_logit:  torch.Tensor | None,
    y_signal:    torch.Tensor,
    y_risk:      torch.Tensor,
    pos_weight:  torch.Tensor,
    risk_lambda: float = RISK_LAMBDA,
    beta:        float = BETA_NLL,
    K:           int   = 1,
    use_crps:    bool  = USE_CRPS_LOSS,
    use_focal:   bool  = USE_FOCAL_LOSS,
) -> tuple[torch.Tensor, float, float]:
    """Combined signal + risk loss.

    Signal component (in priority order):
      1. MDN NLL        if K > 1
      2. Differentiable CRPS  if use_crps (and K == 1 and not deterministic)
      3. Beta-NLL       if probabilistic and not use_crps
      4. MSE            if deterministic

    Risk component:
      Focal BCE         if use_focal
      Weighted BCE      otherwise
    Label smoothing is applied to risk targets.

    Returns (total_loss_tensor, sig_loss_float, risk_loss_float).
    """
    # --- Signal loss ---
    if signal_out.shape[-1] == 1:
        # Deterministic mode
        sig_loss = F.mse_loss(signal_out[..., 0], y_signal)
    elif K > 1:
        # MDN NLL
        sig_loss = mdn_nll_loss(signal_out, y_signal, K)
    elif use_crps:
        # Differentiable CRPS (single Gaussian)
        sig_loss = differentiable_crps_loss(signal_out, y_signal)
    else:
        # Beta-NLL (default)
        sig_loss = gaussian_nll_loss(signal_out, y_signal, beta=beta)

    # --- Risk loss with label smoothing ---
    if risk_logit is not None:
        logit_sq = risk_logit.squeeze(-1)
        y_smooth  = y_risk * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
        if use_focal:
            risk_loss = focal_bce_loss(logit_sq, y_smooth, FOCAL_GAMMA, pos_weight)
        else:
            risk_loss = F.binary_cross_entropy_with_logits(
                logit_sq, y_smooth, pos_weight=pos_weight
            )
        total = sig_loss + risk_lambda * risk_loss
        return total, sig_loss.item(), risk_loss.item()

    return sig_loss, sig_loss.item(), 0.0

# model.py
# Dual-task probabilistic LSTM for ECG forecasting + arrhythmia risk.
#
# The model takes TWO inputs per timestep:
#   1. The raw (normalised) ECG voltage value.
#   2. A beat-type label (0=no beat, 1=normal, 2=abnormal) derived from the
#      MIT-BIH annotation files.  A small learned embedding converts this
#      discrete token into a dense vector that is concatenated with the signal
#      before being fed into the LSTM.
#
# The model produces TWO outputs from the final LSTM hidden state:
#   1. Signal head  — Gaussian distribution over future ECG values:
#                     (batch, forecast_len, 2)  where [...,0]=μ, [...,1]=log σ²
#   2. Risk head    — raw logit for "will an abnormal beat occur in the next
#                     forecast window?":  (batch, 1)
#                     Apply sigmoid to obtain a probability.
#
# Loss
# ----
#   L_total = NLL_signal  +  RISK_LAMBDA * BCE_risk
#
#   NLL_signal  = Gaussian negative log-likelihood (aleatoric uncertainty)
#   BCE_risk    = binary cross-entropy with class-imbalance pos_weight
#
# MC Dropout (called from test.py) keeps dropout active at inference time
# and runs N stochastic forward passes to estimate epistemic uncertainty for
# BOTH heads simultaneously.
#
# GenAI assistance: used to draft the dual-head architecture and combined loss;
# the mathematics were independently verified by the team against Kendall &
# Gal (2017) and standard PyTorch BCE documentation.

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
    FORECAST_LEN, EMBED_DIM, NUM_BEAT_CLASSES, RISK_LAMBDA,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ProbabilisticLSTM(nn.Module):
    """Annotation-aware probabilistic ECG forecaster with arrhythmia risk head.

    Architecture
    ------------
    Input  : x_signal (batch, input_len, 1)    — normalised ECG
             x_annot  (batch, input_len)  int  — beat-type tokens (0/1/2)
    Embed  : Embedding(num_beat_classes, embed_dim)
    Concat : (batch, input_len, 1 + embed_dim)
    LSTM   : multi-layer with inter-layer dropout
    Dropout: applied to the final hidden state
    Heads  :
      fc_mean   → (batch, forecast_len)   predicted μ per timestep
      fc_logvar → (batch, forecast_len)   predicted log σ² per timestep
      fc_risk   → (batch, 1)              raw logit for arrhythmia risk

    Outputs
    -------
    signal_out : (batch, forecast_len, 2)   last dim = [μ, log σ²]
    risk_logit : (batch, 1)                 pass through sigmoid for probability
    """

    def __init__(
        self,
        input_size: int       = 1,
        hidden_size: int      = HIDDEN_SIZE,
        num_layers: int       = NUM_LAYERS,
        forecast_len: int     = FORECAST_LEN,
        dropout: float        = DROPOUT,
        embed_dim: int        = EMBED_DIM,
        num_beat_classes: int = NUM_BEAT_CLASSES,
    ) -> None:
        super().__init__()
        self.forecast_len = forecast_len

        # Learned embedding for beat-type tokens.
        # padding_idx=0 keeps the "no beat" token at the zero vector.
        self.beat_embedding = nn.Embedding(
            num_beat_classes, embed_dim, padding_idx=0
        )

        # LSTM encoder.  Its input size is signal + embedding dimensions.
        # Inter-layer dropout is only active when num_layers > 1.
        self.lstm = nn.LSTM(
            input_size  = input_size + embed_dim,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        # Explicit dropout on the final hidden state — this is the layer that
        # MC Dropout exploits at test time to sample epistemic uncertainty.
        self.dropout = nn.Dropout(dropout)

        # Signal forecasting heads (independent weights → free to learn
        # different scales for mean vs. variance).
        self.fc_mean   = nn.Linear(hidden_size, forecast_len)
        self.fc_logvar = nn.Linear(hidden_size, forecast_len)

        # Arrhythmia risk head (single logit, apply sigmoid externally).
        self.fc_risk = nn.Linear(hidden_size, 1)

    # ------------------------------------------------------------------
    def forward(
        self,
        x_signal: torch.Tensor,  # (batch, input_len, 1)     float32
        x_annot:  torch.Tensor,  # (batch, input_len)        int64
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        signal_out : (batch, forecast_len, 2)   [μ, log σ²]
        risk_logit : (batch, 1)
        """
        # Embed beat tokens and concatenate with raw signal
        emb = self.beat_embedding(x_annot)           # (batch, input_len, embed_dim)
        lstm_in = torch.cat([x_signal, emb], dim=-1) # (batch, input_len, 1+embed_dim)

        # LSTM: take only the output at the final timestep as a sequence summary
        lstm_out, _ = self.lstm(lstm_in)             # (batch, input_len, hidden)
        h = lstm_out[:, -1, :]                       # (batch, hidden)
        h = self.dropout(h)

        # Signal head
        mean   = self.fc_mean(h)                     # (batch, forecast_len)
        logvar = self.fc_logvar(h)                   # (batch, forecast_len)
        logvar = torch.clamp(logvar, -10.0, 10.0)   # numerical stability
        signal_out = torch.stack([mean, logvar], dim=-1)  # (batch, forecast_len, 2)

        # Risk head
        risk_logit = self.fc_risk(h)                 # (batch, 1)

        return signal_out, risk_logit


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def gaussian_nll_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gaussian Negative Log-Likelihood for the signal forecasting head.

    For each predicted timestep:
        NLL_t = 0.5 * (log σ²_t  +  (y_t − μ_t)² / σ²_t)

    Derived from  −log p(y | μ, σ²)  dropping the constant 0.5·log(2π).

    Parameters
    ----------
    output : (batch, forecast_len, 2)   model output  [μ, log σ²]
    target : (batch, forecast_len)      ground-truth future ECG values

    Returns
    -------
    Scalar mean NLL.
    """
    mean   = output[..., 0]
    logvar = output[..., 1]
    precision = torch.exp(-logvar)          # 1/σ²
    nll = 0.5 * (logvar + (target - mean) ** 2 * precision)
    return nll.mean()


def combined_loss(
    signal_out:  torch.Tensor,  # (batch, forecast_len, 2)
    risk_logit:  torch.Tensor,  # (batch, 1)
    y_signal:    torch.Tensor,  # (batch, forecast_len)
    y_risk:      torch.Tensor,  # (batch,)  float  0/1
    pos_weight:  torch.Tensor,  # scalar tensor — BCE imbalance weight
    risk_lambda: float = RISK_LAMBDA,
) -> tuple[torch.Tensor, float, float]:
    """Combined signal NLL + arrhythmia risk BCE loss.

    Parameters
    ----------
    pos_weight : tensor(neg_count / pos_count) on the correct device

    Returns
    -------
    total_loss  : differentiable scalar
    nll_val     : float  (detached, for logging)
    risk_val    : float  (detached, for logging)
    """
    nll  = gaussian_nll_loss(signal_out, y_signal)
    risk = F.binary_cross_entropy_with_logits(
        risk_logit.squeeze(-1),
        y_risk,
        pos_weight=pos_weight,
    )
    total = nll + risk_lambda * risk
    return total, nll.item(), risk.item()

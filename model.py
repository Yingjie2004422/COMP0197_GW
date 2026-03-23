# model.py
# Probabilistic LSTM model for ECG time-series forecasting.
#
# The model outputs a Gaussian distribution over future ECG values by
# predicting a *mean* (μ) and a *log-variance* (log σ²) for every forecast
# timestep.  This is a standard approach to aleatoric (data) uncertainty.
#
# MC Dropout (called from test.py) additionally captures epistemic
# (model) uncertainty by keeping dropout active at inference time and running
# multiple stochastic forward passes.
#
# GenAI assistance: used to draft the dual-head architecture; the mathematics
# of the Gaussian NLL loss were independently verified by the team.

import torch
import torch.nn as nn

from config import HIDDEN_SIZE, NUM_LAYERS, DROPOUT, FORECAST_LEN


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ProbabilisticLSTM(nn.Module):
    """LSTM-based probabilistic ECG forecaster.

    Architecture
    ------------
    Input  : (batch, input_len, 1)
    LSTM   : multi-layer with dropout between layers
    Heads  : two parallel Linear layers on the final hidden state
               fc_mean   → (batch, forecast_len)   predicted μ per timestep
               fc_logvar → (batch, forecast_len)   predicted log σ² per timestep
    Output : (batch, forecast_len, 2)  last dim = [μ, log σ²]

    The log-variance is clamped to [-10, 10] for numerical stability before
    being returned.  Callers should use exp(log σ²) to obtain variance and
    must NOT apply any further activation themselves.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        forecast_len: int = FORECAST_LEN,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.forecast_len = forecast_len

        # LSTM encoder — dropout is only inserted *between* layers, so it has
        # no effect when num_layers == 1.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Explicit dropout applied to the last hidden state before the heads.
        # This is the dropout that MC Dropout exploits at test time.
        self.dropout = nn.Dropout(dropout)

        # Two separate output heads share no parameters so that mean and
        # log-variance can be learned independently.
        self.fc_mean   = nn.Linear(hidden_size, forecast_len)
        self.fc_logvar = nn.Linear(hidden_size, forecast_len)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, input_len, 1)  — normalised ECG input window

        Returns
        -------
        out : (batch, forecast_len, 2)
              out[..., 0] = μ        (mean prediction)
              out[..., 1] = log σ²   (log variance, clamped to [-10, 10])
        """
        # Run LSTM; use only the output at the final timestep as a summary
        lstm_out, _ = self.lstm(x)          # (batch, input_len, hidden)
        h = lstm_out[:, -1, :]              # (batch, hidden)
        h = self.dropout(h)

        mean   = self.fc_mean(h)            # (batch, forecast_len)
        logvar = self.fc_logvar(h)          # (batch, forecast_len)
        logvar = torch.clamp(logvar, -10.0, 10.0)

        # Stack along a new last dimension → (batch, forecast_len, 2)
        return torch.stack([mean, logvar], dim=-1)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def gaussian_nll_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gaussian Negative Log-Likelihood loss.

    For each predicted timestep the loss is:

        NLL_t = 0.5 * (log σ²_t  +  (y_t - μ_t)² / σ²_t)

    This is derived from  -log p(y | μ, σ²)  with a unit Gaussian, dropping
    the constant 0.5 * log(2π) term (irrelevant for optimisation).

    Parameters
    ----------
    output : (batch, forecast_len, 2)  model output  [μ, log σ²]
    target : (batch, forecast_len)     ground-truth future ECG values

    Returns
    -------
    Scalar mean NLL loss.
    """
    mean   = output[..., 0]          # (batch, forecast_len)
    logvar = output[..., 1]          # (batch, forecast_len)

    # precision = 1 / σ²  =  exp(-log σ²)
    precision = torch.exp(-logvar)
    nll = 0.5 * (logvar + (target - mean) ** 2 * precision)

    return nll.mean()

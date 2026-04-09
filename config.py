# config.py
# Central configuration for the ECG probabilistic forecasting project.
# All hyperparameters and paths are defined here so that every other file
# imports from a single source of truth.
#
# --- Ablation study guide ---
# Change these flags and re-run train.py to produce each ablation model:
#
#   Full model (default):    USE_RR_FEATURES=True,  USE_RISK_HEAD=True,  DETERMINISTIC=False
#   No attention:            USE_ATTENTION=False
#   No RR features:          USE_RR_FEATURES=False, USE_RISK_HEAD=True,  DETERMINISTIC=False
#   No risk head:            USE_RR_FEATURES=True,  USE_RISK_HEAD=False, DETERMINISTIC=False
#   Deterministic baseline:  USE_RR_FEATURES=True,  USE_RISK_HEAD=False, DETERMINISTIC=True
#
# GenAI assistance: used to scaffold the initial structure; all values and
# design decisions were reviewed and validated by the team.
# Improvements: bidirectional LSTM, multi-head attention, seq2seq decoder,
# dual-lead input, HRV features, cosine warm restarts, label smoothing,
# temperature scaling, conformal risk prediction, Diebold-Mariano test.

# --- Paths ---
DATA_FOLDER = "mitdb"
MODEL_DIR   = "checkpoints"
RESULTS_DIR = "results"

# --- Signal parameters ---
SAMPLING_RATE = 360         # MIT-BIH records are sampled at 360 Hz
INPUT_LEN     = 720         # 2 seconds of past ECG + annotations fed to the model
FORECAST_LEN  = 180         # 0.5 seconds of future ECG to predict
STRIDE        = 180         # 50% window overlap; balance diversity vs dataset size

# --- Training ---
BATCH_SIZE      = 64
NUM_EPOCHS      = 60        # upper limit; early stopping usually terminates earlier
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4      # L2 regularisation via AdamW (decoupled weight decay)
TRAIN_VAL_SPLIT = 0.8       # fraction of records used for training
SEED            = 42

# --- Early stopping ---
EARLY_STOPPING_PATIENCE = 8  # halt if val loss does not improve for N consecutive epochs

# --- Data augmentation ---
AUGMENT_TRAIN = True   # apply random noise / scaling / wander to training batches

# --- Model architecture ---
HIDDEN_SIZE    = 128
NUM_LAYERS     = 2
DROPOUT        = 0.3        # applied between LSTM layers and before output heads
USE_LAYER_NORM = True       # LayerNorm on the final LSTM hidden state

# --- Temporal attention ---
# Multi-head attention over all LSTM output timesteps instead of only using the last hidden state.
# Allows the model to focus on the most diagnostically relevant input beats.
USE_ATTENTION  = True
N_ATTN_HEADS   = 4          # number of attention heads for MultiHeadTemporalAttention

# --- Beat-annotation embedding ---
NUM_BEAT_CLASSES = 3
EMBED_DIM        = 8

# --- Ablation / feature flags ---
USE_RR_FEATURES = True   # include normalised RR + dRR as 2 extra input channels
USE_RISK_HEAD   = True   # add arrhythmia risk binary classification head
DETERMINISTIC   = False  # True → output mean only and train with MSE (baseline)

# --- Combined loss ---
RISK_LAMBDA = 1.0

# --- Beta-NLL (Seitzer et al., 2022) ---
# Weights the NLL loss by σ^(2β) to prevent variance collapse.
# β=0 → standard NLL; β=0.5 → recommended default.
BETA_NLL = 0.5

# --- MC Dropout inference (epistemic uncertainty) ---
MC_SAMPLES = 50

# --- Mixture Density Network ---
# USE_MDN=True: K_MDN Gaussian components instead of a single Gaussian.
# K=1 keeps the original single-Gaussian behaviour for backward compatibility.
USE_MDN = True
K_MDN   = 3

# --- Deep Ensemble ---
# Train N_ENSEMBLE independent models with different random seeds.
# At inference time their predictions are combined to obtain both aleatoric
# and epistemic uncertainty estimates.
N_ENSEMBLE = 3

# --- Focal loss for risk head ---
# Focal loss (Lin et al., 2017) down-weights easy negatives and focuses
# training on hard positives.  gamma=2.0 is the standard recommendation.
USE_FOCAL_LOSS = True
FOCAL_GAMMA    = 2.0

# --- CRPS as training objective ---
# If True, use the differentiable closed-form Gaussian CRPS as the signal
# training loss instead of Beta-NLL.  Disabled by default.
USE_CRPS_LOSS = False

# --- Curriculum learning ---
# For the first CURRICULUM_EPOCHS epochs, train only on windows that contain
# no arrhythmia (normal_loader).  After that, switch to the full train_loader.
CURRICULUM_EPOCHS = 5

# --- Arrhythmia window oversampling ---
# Use WeightedRandomSampler so arrhythmia windows are sampled at a higher
# rate during training, mitigating severe class imbalance.
USE_OVERSAMPLING = True

# --- Beat morphology feature channels ---
# 8 channels: [RR, dRR, peak_amp, dAmp, RR_mask, dRR_mask, amp_mask, dAmp_mask]
# Previously 4 channels: [RR, dRR, peak_amp, dAmp]; added mask channels
# Previously 2 channels (RR, dRR); peak_amp and dAmp are added here.
N_FEAT_CHANNELS = 8   # was implicitly 2, then 4 (RR, dRR, peak_amp, dAmp), and now 8 (RR, dRR, peak_amp, dAmp, RR_mask, dRR_mask, amp_mask, dAmp_mask)

# --- Dual-lead input ---
# Number of ECG signal input channels (leads). 2 = lead 0 + lead 1.
INPUT_CHANNELS = 2

# --- HRV features ---
# Number of HRV scalar features computed per input window: SDNN, RMSSD, pNN50.
N_HRV_FEATURES = 3

# --- Seq2Seq decoder teacher forcing (scheduled) ---
# Linearly annealed from TEACHER_FORCING_START (epoch 1) to TEACHER_FORCING_END
# (final epoch).  High ratio early = stable gradient signal; low ratio late =
# reduces exposure bias so the model learns to chain its own predictions.
TEACHER_FORCING_RATIO = 0.5   # default / fallback (overridden by schedule)
TEACHER_FORCING_START = 0.90
TEACHER_FORCING_END   = 0.10

# --- Label smoothing for risk head ---
# Soft targets: y_smooth = y * (1 - eps) + 0.5 * eps
LABEL_SMOOTHING = 0.05

# --- LR scheduler: CosineAnnealingWarmRestarts ---
LR_T0     = 15   # number of epochs for first restart
LR_T_MULT = 2    # factor by which T_0 increases after each restart

# --- Conformal prediction for risk head ---
CONFORMAL_ALPHA_RISK = 0.10

# config.py
# Central configuration for the ECG probabilistic forecasting project.
# All hyperparameters and paths are defined here.
#
# GenAI assistance: used to scaffold the initial structure; all values and
# design decisions were reviewed and validated by the team.

# --- Paths ---
DATA_FOLDER = "mitdb"       # folder containing MIT-BIH .dat/.hea/.atr files
MODEL_DIR   = "checkpoints" # where best_model.pt is saved during training
RESULTS_DIR = "results"     # where plots are written by test.py

# --- Signal parameters ---
SAMPLING_RATE = 360         # MIT-BIH records are sampled at 360 Hz

# Sliding-window sizes (in samples).
# INPUT_LEN  = 1 second  (360 samples) of past ECG signal given to the model
# FORECAST_LEN = 0.5 s   (180 samples) of future ECG signal to be predicted
INPUT_LEN    = 360
FORECAST_LEN = 180

# Step between consecutive windows.
# STRIDE = 180 means windows overlap by 50 %, keeping diversity while
# preventing the dataset from becoming prohibitively large.
STRIDE = 180

# --- Training ---
BATCH_SIZE      = 64
NUM_EPOCHS      = 30
LEARNING_RATE   = 1e-3
TRAIN_VAL_SPLIT = 0.8   # fraction of records used for training (rest = val)
SEED            = 42

# --- Model architecture (ProbabilisticLSTM) ---
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.3       # applied between LSTM layers and before output heads

# --- MC Dropout inference (epistemic uncertainty) ---
# Number of stochastic forward passes at test time.
MC_SAMPLES = 50

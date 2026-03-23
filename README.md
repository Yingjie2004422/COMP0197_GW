# COMP0197 Group Coursework — Probabilistic ECG Forecasting

Deep learning system that predicts future ECG signal values as a **probability distribution**, quantifying both aleatoric (data) and epistemic (model) uncertainty using the MIT-BIH Arrhythmia Dataset.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```
Saves the best checkpoint to `checkpoints/best_model.pt` and a loss curve to `results/loss_curves.png`.

### 3. Evaluate and generate plots
```bash
python test.py
```
Reports MSE and NLL on the validation set and saves prediction + uncertainty plots to `results/`.

---

## Project Structure

```
COMP0197_GW/
├── config.py      # All hyperparameters and paths (edit here to change settings)
├── dataset.py     # ECGDataset and DataLoader factory
├── model.py       # ProbabilisticLSTM + Gaussian NLL loss
├── train.py       # Training loop with checkpointing
├── test.py        # Evaluation, prediction plots, uncertainty analysis
├── checkpoints/   # Saved model weights (best_model.pt)
├── mitdb/         # MIT-BIH Arrhythmia Database records
└── results/       # Generated plots (created by test.py)
```

---

## Dataset

The `mitdb/` folder contains the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) (48 records, 360 Hz).
It is included in this repository. If it is missing, download it with:

```python
import wfdb
wfdb.dl_database('mitdb', dl_dir='mitdb')
```

---

## Key Design Choices

| Component | Choice | Reason |
|---|---|---|
| Model | LSTM (2 layers, hidden=128) | Effective for temporal sequences; interpretable |
| Uncertainty | Gaussian NLL (aleatoric) + MC Dropout (epistemic) | Direct probabilistic output without ensembles |
| Loss | Gaussian Negative Log-Likelihood | Trains μ and σ² jointly; principled Bayesian objective |
| Data split | Record-level 80/20 | Prevents temporal leakage between train and val |

---

## Configuration

All settings are in [`config.py`](config.py). Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `INPUT_LEN` | 360 | Input window size (1 second at 360 Hz) |
| `FORECAST_LEN` | 180 | Forecast horizon (0.5 seconds) |
| `STRIDE` | 180 | Step between sliding windows |
| `NUM_EPOCHS` | 30 | Training epochs |
| `MC_SAMPLES` | 50 | MC Dropout passes for epistemic uncertainty |

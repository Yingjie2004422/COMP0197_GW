# data.py
# Data loading and preprocessing for ECG forecasting cw
import os
import wfdb
import torch
from torch.utils.data import TensorDataset, DataLoader
from config import DATA_FOLDER


def load_all_records(data_folder=DATA_FOLDER):
    """
    Loads all .dat/.hea records in the folder, normalizes ECG, and returns a list of tensors.
    """
    # Get all record names (without extension)
    records = sorted(list(set(f.split('.')[0] for f in os.listdir(data_folder) if f.endswith('.dat'))))

    all_ecg = []
    for r in records:
        record_path = os.path.join(data_folder, r)
        record = wfdb.rdrecord(record_path)
        ecg = record.p_signal[:, 0] 
        ecg = torch.tensor(ecg, dtype=torch.float32)
        # Normalize
        ecg = (ecg - ecg.mean()) / ecg.std()
        all_ecg.append(ecg)

    return all_ecg


def create_windows(signal, input_len=200, forecast_len=50):
    """
    Converts a long 1D ECG signal into input-output sliding windows for forecasting.
    """
    X, Y = [], []
    for i in range(len(signal) - input_len - forecast_len):
        x = signal[i:i+input_len]
        y = signal[i+input_len:i+input_len+forecast_len]
        X.append(x)
        Y.append(y)

    X = torch.stack(X).unsqueeze(1)  # (num_samples, 1, input_len)
    Y = torch.stack(Y).unsqueeze(1)  # (num_samples, 1, forecast_len)
    return X, Y


def get_dataloader(data_folder=DATA_FOLDER, input_len=200, forecast_len=50, batch_size=32):
    """
    Returns a PyTorch DataLoader for all ECG records in the folder.
    """
    all_ecg = load_all_records(data_folder)
    all_X, all_Y = [], []

    for ecg in all_ecg:
        X, Y = create_windows(ecg, input_len, forecast_len)
        all_X.append(X)
        all_Y.append(Y)

    all_X = torch.cat(all_X, dim=0)
    all_Y = torch.cat(all_Y, dim=0)

    dataset = TensorDataset(all_X, all_Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
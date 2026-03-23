# model.py
import torch
import torch.nn as nn
from config import INPUT_LEN, FORECAST_LEN



# ........ skeleton of the model, you can modify it as needed ........
class ECGForecaster(nn.Module):
    def __init__(self, input_len=INPUT_LEN, forecast_len=FORECAST_LEN):
        super(ECGForecaster, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.fc = nn.Linear(64 * (input_len // 8), forecast_len)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool1d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool1d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool1d(x, 2)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


# train.py
from data import get_dataloader
from config import DATA_FOLDER, INPUT_LEN, FORECAST_LEN, BATCH_SIZE

dataloader = get_dataloader(
    data_folder=DATA_FOLDER, 
    input_len=INPUT_LEN, 
    forecast_len=FORECAST_LEN, 
    batch_size=BATCH_SIZE
)

# test the dataloader
x, y = next(iter(dataloader))
print(x.shape, y.shape)
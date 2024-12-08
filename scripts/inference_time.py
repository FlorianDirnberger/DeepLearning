import time
import torch
from torch.utils.data import DataLoader
from models import RNN  # Adjust import as needed
from modular_train_test_rnn import TimeSeriesDataset, DATA_ROOT, DATA_SUBDIR, STM_FILENAME, FIXED_LENGTH, DEVICE

# Load dataset
data_dir = DATA_ROOT / DATA_SUBDIR / "test"
stmf_data_path = DATA_ROOT / STM_FILENAME
test_dataset = TimeSeriesDataset(data_dir, stmf_data_path, fixed_length=FIXED_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Function to calculate inference time
def evaluate_inference_time(model, data_loader):
    model.eval()
    total_time = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            timeseries = batch["timeseries"].to(DEVICE)
            
            # Measure inference time
            start_time = time.perf_counter()
            outputs = model(timeseries)
            end_time = time.perf_counter()
            
            inference_time = end_time - start_time
            total_time += inference_time
            num_samples += timeseries.size(0)

    avg_inference_time = total_time / num_samples
    return avg_inference_time

# Function to count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Evaluate all models
model_configs = {
    "RNN": {"mode": "RNN", "hidden_size": 256, "num_layers": 2, "dropout": 0.3},
    "GRU": {"mode": "GRU", "hidden_size": 256, "num_layers": 2, "dropout": 0},
    "LSTM": {"mode": "LSTM", "hidden_size": 128, "num_layers": 8, "dropout": 0.3},
}

for model_name, config in model_configs.items():
    # Load model
    model = RNN(
        mode=config["mode"],
        input_size=8,  # Adjust according to your dataset
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        output_size=1,
    ).to(DEVICE)

    # Load pre-trained weights
    model_path = f"models/{model_name}_best.pth"  # Adjust path as needed
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # Count model parameters
    num_params = count_parameters(model)

    # Evaluate inference time
    avg_time = evaluate_inference_time(model, test_loader)

    # Print results
    print(f"{model_name} - Parameters: {num_params:,}, Average Inference Time per Sample: {avg_time:.6f} seconds")

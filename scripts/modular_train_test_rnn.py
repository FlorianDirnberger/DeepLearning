from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import wandb
import torch.nn as nn
from dotenv import load_dotenv
import os
import pickle
import pandas as pd

# GROUP NUMBER
GROUP_NUMBER = 97

# Adjusted Constants
DATA_ROOT = Path("/dtu-compute/02456-p4-e24/data")
DATA_SUBDIR = "data_fft-512_tscropwidth-150-200_vrcropwidth-60-15"
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
STM_FILENAME = "stmf_data_3.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

# Load environment variables from .env file
load_dotenv()
# Use the API key from the environment variable
wandb_api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=wandb_api_key)


class TimeSeriesDataset(Dataset):
    def __init__(self, data_dir, stmf_data_path, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Load the ground truth labels
        self.stmf_df = pd.read_csv(stmf_data_path)

        # Extract IDs and labels
        self.sample_ids = self.stmf_df.iloc[:, 0].tolist()
        self.labels = self.stmf_df['BallVr'].tolist()

        # Exclude missing files
        valid_ids = []
        valid_labels = []
        for sample_id, label in zip(self.sample_ids, self.labels):
            data_path = self.data_dir / f"{sample_id}_timeseries.pkl"
            if data_path.exists():
                valid_ids.append(sample_id)
                valid_labels.append(label)
        self.sample_ids = valid_ids
        self.labels = valid_labels

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        label = self.labels[idx]

        # Build the filename for the .pkl file
        data_path = self.data_dir / f"{sample_id}_timeseries.pkl"

        # Load the time series data from the .pkl file
        with open(data_path, 'rb') as f:
            timeseries_data = pickle.load(f)

        # Apply any transforms if specified
        if self.transform:
            timeseries_data = self.transform(timeseries_data)

        # Convert timeseries_data and label to tensors
        timeseries_data = torch.tensor(timeseries_data, dtype=torch.float32)

        # Ensure timeseries_data has shape (seq_len, input_size)
        if timeseries_data.dim() == 1:
            timeseries_data = timeseries_data.unsqueeze(-1)  # Add input_size dimension

        label = torch.tensor(label, dtype=torch.float32)
        return {'timeseries': timeseries_data, 'label': label}



def train_one_epoch(loss_fn, model, train_data_loader, optimizer):
    running_loss = 0.0
    total_loss = 0.0

    for i, data in enumerate(train_data_loader):
        timeseries, label = data["timeseries"].to(DEVICE), data["label"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(timeseries)

        # Compute the loss and its gradients
        loss = loss_fn(outputs.squeeze(), label)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        total_loss += loss.item()
        if (i + 1) % train_data_loader.batch_size == 0:
            last_loss = running_loss / train_data_loader.batch_size
            print(f'  batch {i + 1} loss: {last_loss}')
            running_loss = 0.0

    return total_loss / (i + 1)


def train():
    with wandb.init() as run:
        config = run.config

        # Data Setup
        data_dir = DATA_ROOT / DATA_SUBDIR
        stmf_data_path = DATA_ROOT / STM_FILENAME

        print(f"Data root directory: {DATA_ROOT}")
        print(f"Data subdirectory: {data_dir}")
        print(f"Train data directory: {data_dir / 'train'}")
        print(f"Test data directory: {data_dir / 'test'}")
        print(f"STM filename path: {stmf_data_path}")

        dataset_train = TimeSeriesDataset(
            data_dir=data_dir / "train",
            stmf_data_path=stmf_data_path
        )
        dataset_test = TimeSeriesDataset(
            data_dir=data_dir / "test",
            stmf_data_path=stmf_data_path
        )

        train_data_loader = DataLoader(
            dataset_train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4
        )
        test_data_loader = DataLoader(
            dataset_test,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=1
        )

        # Determine input_size based on your data
        sample = next(iter(train_data_loader))
        timeseries_sample = sample['timeseries']
        input_size = timeseries_sample.shape[2]  # timeseries_sample shape: (batch_size, seq_len, input_size)

        # Initialize model with the current WandB configuration
        model = RNN(
            input_size=input_size,
            hidden_size=config.hidden_size,
            output_size=1,
            num_layers=config.num_layers
        ).to(DEVICE)

        # Initialize weights if necessary
        model.apply(weights_init_uniform_rule)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
        loss_fn = nn.MSELoss()

        # Training Loop
        best_vloss = float('inf')
        exceed_rmse_count = 0  # Counter for consecutive epochs with test_rmse > 8
        for epoch in range(config.epochs):
            print(f'EPOCH {epoch + 1}:')
            model.train(True)
            avg_loss = train_one_epoch(loss_fn, model, train_data_loader, optimizer)

            rmse = avg_loss ** 0.5

            running_test_loss = 0.0
            model.eval()
            with torch.no_grad():
                for i, vdata in enumerate(test_data_loader):
                    timeseries, label = vdata["timeseries"].to(DEVICE), vdata["label"].to(DEVICE)
                    test_outputs = model(timeseries)
                    test_loss = loss_fn(test_outputs.squeeze(), label)
                    running_test_loss += test_loss.item()

            avg_test_loss = running_test_loss / (i + 1)
            test_rmse = avg_test_loss ** 0.5

            print(f'LOSS train {avg_loss} ; LOSS test {avg_test_loss}')

            # Log metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "rmse": rmse,
                "test_loss": avg_test_loss,
                "test_rmse": test_rmse,
            })

            # Track best performance, and save the model's state
            if avg_test_loss < best_vloss:
                best_vloss = avg_test_loss
                model_path = MODEL_DIR / f"model_{model.__class__.__name__}_{wandb.run.name}.pt"
                torch.save(model.state_dict(), model_path)

            if test_rmse > 8:
                exceed_rmse_count += 1
                if exceed_rmse_count >= 10:
                    print("Test RMSE exceeded 8 for 10 consecutive epochs. Ending training early.")
                    break
            else:
                exceed_rmse_count = 0  # Reset counter if condition is not met


# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Get the output from the last time step
        out = self.fc(out)
        return out


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # For LSTM and Linear layers
    if classname.find('Linear') != -1 or classname.find('LSTM') != -1:
        # Initialize weights with uniform distribution
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.uniform_(m.weight.data, -0.1, 0.1)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


# WandB Sweep Configuration
sweep_config = {
    'method': 'random',  # Specifies random search

    'metric':
        {'name': 'test_rmse', 'goal': 'minimize'},

    'parameters': {
        'hidden_size': {
            'values': [64, 128, 256]
        },
        'num_layers': {
            'values': [1, 2, 3]
        },
        'learning_rate': {
            'values': [1e-3, 1e-4, 1e-5]
        },
        'epochs': {
            'values': [20]
        },
        'batch_size': {
            'values': [16, 32, 64]
        }
    }
}

if __name__ == "__main__":
    # Initialize the sweep in WandB
    sweep_id = wandb.sweep(sweep_config, project="Deep Learning Project Group 97")
    wandb.agent(sweep_id, function=train)

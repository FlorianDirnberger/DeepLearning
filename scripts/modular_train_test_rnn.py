from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import wandb
import torch.nn as nn
from dotenv import load_dotenv
import os
import pickle
import pandas as pd
import numpy as np

# Constants and Configurations
GROUP_NUMBER = 97
DATA_ROOT = Path("/dtu-compute/02456-p4-e24/data")
DATA_SUBDIR = "data_fft-512_tscropwidth-150-200_vrcropwidth-60-15"
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
STM_FILENAME = "stmf_data_3.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIXED_LENGTH = 1600  # Fixed length for all sequences

# Load environment variables
load_dotenv()
wandb_api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=wandb_api_key)


class TimeSeriesDataset(Dataset):
    def __init__(self, data_dir, stmf_data_path, fixed_length=1600, transform=None):
        self.data_dir = data_dir
        self.fixed_length = fixed_length
        self.transform = transform

        # Load ground truth labels
        self.stmf_df = pd.read_csv(stmf_data_path)
        self.sample_ids = self.stmf_df.iloc[:, 0].tolist()
        self.labels = self.stmf_df['BallVr'].tolist()

        # Filter valid `.pkl` files
        available_pkl_files = {f.stem.split('_')[0] for f in data_dir.glob("*_timeseries.pkl")}
        self.sample_ids, self.labels = zip(*[
            (sample_id, label)
            for sample_id, label in zip(self.sample_ids, self.labels)
            if str(sample_id) in available_pkl_files
        ])

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        label = self.labels[idx]
        data_path = self.data_dir / f"{sample_id}_timeseries.pkl"

        if not data_path.exists():
            raise FileNotFoundError(f"Missing .pkl file: {data_path}")

        try:
            with open(data_path, 'rb') as f:
                timeseries_data = pickle.load(f)
                if isinstance(timeseries_data, dict) and 'samples' in timeseries_data:
                    samples = timeseries_data['samples']
                else:
                    raise ValueError(f"Invalid file format for {data_path}")

                # Ensure the data is a numpy array
                if not isinstance(samples, np.ndarray):
                    samples = np.array(samples)

                # Convert complex-valued data to real-imaginary representation
                if np.iscomplexobj(samples):
                    samples = np.stack([samples.real, samples.imag], axis=-1)

                # Pad or truncate to fixed length
                if samples.shape[0] > self.fixed_length:
                    samples = samples[:self.fixed_length]  # Truncate
                elif samples.shape[0] < self.fixed_length:
                    padding = self.fixed_length - samples.shape[0]
                    samples = np.pad(samples, ((0, padding), (0, 0)), mode='constant')  # Pad

        except Exception as e:
            print(f"Error reading file {data_path}: {e}")
            raise

        if self.transform:
            samples = self.transform(samples)

        return {
            'timeseries': torch.tensor(samples, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Get the output from the last time step
        return self.fc(out)


def train_one_epoch(loss_fn, model, data_loader, optimizer):
    model.train()
    running_loss = 0.0

    for batch in data_loader:
        timeseries, labels = batch["timeseries"].to(DEVICE), batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(timeseries)
        loss = loss_fn(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(data_loader)


def train():
    with wandb.init(project="DeepLearning-scripts", config={
        "batch_size": 16,
        "hidden_size": 128,
        "num_layers": 2,
        "learning_rate": 0.01,
        "epochs": 10
    }) as run:
        config = run.config

        data_dir = DATA_ROOT / DATA_SUBDIR
        stmf_data_path = DATA_ROOT / STM_FILENAME
        dataset_train = TimeSeriesDataset(data_dir / "train", stmf_data_path, fixed_length=FIXED_LENGTH)
        dataset_test = TimeSeriesDataset(data_dir / "test", stmf_data_path, fixed_length=FIXED_LENGTH)

        train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False)

        # Determine input size
        sample = next(iter(train_loader))
        input_size = sample['timeseries'].shape[-1]

        model = RNN(
            input_size=input_size,
            hidden_size=config.hidden_size,
            output_size=1,
            num_layers=config.num_layers
        ).to(DEVICE)
        model.apply(weights_init_uniform_rule)

        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
        loss_fn = nn.MSELoss()

        print(f"Starting training for {config.epochs} epochs...")

        for epoch in range(config.epochs):
            train_loss = train_one_epoch(loss_fn, model, train_loader, optimizer)
            print(f"Epoch {epoch + 1}/{config.epochs}, Train Loss: {train_loss:.4f}")

            wandb.log({"epoch": epoch + 1, "train_loss": train_loss})

        torch.save(model.state_dict(), MODEL_DIR / f"RNN_model_{wandb.run.name}.pth")
        print("Training complete. Model saved.")


def weights_init_uniform_rule(m):
    if isinstance(m, (nn.Linear, nn.LSTM)):
        nn.init.uniform_(m.weight.data, -0.1, 0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    test_dataset = False

    if test_dataset:
        data_dir = DATA_ROOT / DATA_SUBDIR / "train"
        stmf_data_path = DATA_ROOT / STM_FILENAME
        dataset = TimeSeriesDataset(data_dir, stmf_data_path, fixed_length=FIXED_LENGTH)

        print(f"Number of valid samples: {len(dataset)}")
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            print(f"Sample {i}: Timeseries shape: {sample['timeseries'].shape}, Label: {sample['label']}")
    else:
        train()

import os
from pathlib import Path
from dotenv import load_dotenv
from numpy import log10
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt
from data_management import make_dataset_name
from Features import LoadSpectrogram,ProcessSpectrogram,InterpolateSpectrogram,ToTensor,process_spectrogram
from torch.utils.data import Dataset
from data_management import get_observation_nums
from typing import Iterable
import pandas as pd

class SpectrogramDataset(Dataset):
    def __init__(self,
                 transform,
                 stmf_data_path: Path,
                 data_dir: Path = None,
                 observation_nums: Iterable = None,
                 csv_delimiter: str = ",") -> None:
        
        if data_dir is None and observation_nums is None:
            raise ValueError("Either `data_dir` or `observation_nums` mus be different from None")
        
        if data_dir is not None:
            observation_nums = get_observation_nums(data_dir)
        self.observation_nums = observation_nums

        self.stmf_data  = pd.read_csv(stmf_data_path, delimiter=csv_delimiter).iloc[observation_nums]
        self.targets = self.stmf_data.BallVr.to_numpy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.targets)
    
    def __getitem__(self, idx: int) -> dict:
        spectrogram, target = self._get_item_helper(idx)
        sample = {"spectrogram": spectrogram, "target": target}

        return sample
    
    def _get_item_helper(self, idx: int) -> tuple:
        stmf_row = self.stmf_data.iloc[idx]

        # Transform the spectrogram row into a NumPy array
        spectrogram = self.transform(stmf_row)

        # Ensure the `process_spectrogram` extracts features
        feature_vector = process_spectrogram(spectrogram)  # Extract features
        feature_vector = torch.tensor(feature_vector, dtype=torch.float32)  # Convert to tensor

        # Target remains unchanged
        target = self.targets[idx]
        target = torch.tensor(target, dtype=torch.float32)

        return feature_vector, target






class FeatureANN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(FeatureANN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# GROUP NUMBER
GROUP_NUMBER = 97
# CONSTANTS TO MODIFY AS YOU WISH
MODEL=FeatureANN
EPOCHS = 300 # the model converges in test perfermance after ~250-300 epochs
BATCH_SIZE = 10
NUM_WORKERS = 4
#OPTIMIZER = torch.optim.SGD
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# DEVICE = "cpu"
# Load environment variables from .env file
load_dotenv()

# Use the API key from the environment variable
# wandb_api_key = os.getenv("WANDB_API_KEY")
# wandb.login(key=wandb_api_key)


# You can set the model path name in case you want to keep training it.
# During the training/testing loop, the model state is saved
# (only the best model so far is saved)
LOAD_MODEL_FNAME = None

# CONSTANTS TO LEAVE
DATA_ROOT = Path(f"/dtu-compute/02456-p4-e24/data") 
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
STMF_FILENAME = "stmf_data_3.csv"
NFFT = 512
TS_CROPTWIDTH = (-150, 80)
VR_CROPTWIDTH = (-60, 0)



# Define a training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)  # MSE loss
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

# Define a testing function
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)  # Predicted velocity
            loss = criterion(outputs.squeeze(), labels)  # MSE loss

            running_loss += loss.item()

    return running_loss / len(test_loader)

if __name__ == "__main__":
    print(f"Using {DEVICE} device")

    # DATA SET SETUP
    dataset_name = make_dataset_name(nfft=NFFT, ts_crop_width=TS_CROPTWIDTH, vr_crop_width=VR_CROPTWIDTH)
    data_dir = DATA_ROOT / dataset_name

    TRAIN_TRANSFORM = transforms.Compose(
    [
        LoadSpectrogram(root_dir=data_dir / "train"),
        ToTensor(),
        InterpolateSpectrogram()
    ]
    )

    TEST_TRANSFORM = transforms.Compose(
    [
            LoadSpectrogram(root_dir=data_dir / "test"),                       
           ToTensor(),
            InterpolateSpectrogram()
    ]
    )

    dataset_train = SpectrogramDataset(
        transform=TRAIN_TRANSFORM,
        stmf_data_path=DATA_ROOT / STMF_FILENAME,
        data_dir=data_dir / "train"
    )

    dataset_test = SpectrogramDataset(
        transform=TEST_TRANSFORM,
        stmf_data_path=DATA_ROOT / STMF_FILENAME,
        data_dir=data_dir / "test"
    )

    train_data_loader = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    test_data_loader = DataLoader(
        dataset_test,
        batch_size=500,
        shuffle=False,
        num_workers=1
    )

    # Model initialization
    model = FeatureANN(input_size=52, output_size=1).to(DEVICE)
    criterion = torch.nn.MSELoss()  # Use an appropriate loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

    best_accuracy = 0.0

    for epoch in range(EPOCHS):
        train_loss = train(model, train_data_loader, criterion, optimizer, DEVICE)
        test_loss, test_accuracy = test(model, test_data_loader, criterion, DEVICE)

        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.2f}%")

    
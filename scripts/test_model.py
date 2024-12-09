import torch
import torch.nn as nn
from torchvision.transforms import transforms

from torch.utils.data import DataLoader
from custom_transforms import LoadSpectrogram, NormalizeSpectrogram, ToTensor, InterpolateSpectrogram
from data_management import make_dataset_name
from pathlib import Path

from models import CNN_97

import time

# CONSTANTS TO LEAVE
DATA_ROOT = Path(f"/dtu-compute/02456-p4-e24/data") 
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
STMF_FILENAME = "stmf_data_3.csv"
NFFT = 512
TS_CROPTWIDTH = (-150, 200)
VR_CROPTWIDTH = (-60, 15)
DEVICE = "cpu"

# Data Setup
dataset_name = make_dataset_name(nfft=NFFT, ts_crop_width=TS_CROPTWIDTH, vr_crop_width=VR_CROPTWIDTH)
data_dir = DATA_ROOT / dataset_name
VAL_TRANSFORM = transforms.Compose([
            LoadSpectrogram(root_dir=data_dir / "validation"),
            NormalizeSpectrogram(),
            ToTensor(),
            InterpolateSpectrogram()
        ])


# Define model with parameters used in the model considering
model = CNN_97(
    num_conv_layers=3,  # Example value, use the same as the trained model
    num_fc_layers=3,
    conv_dropout=0.2,
    linear_dropout=0,
    kernel_size=(5, 7),
    activation=nn.ReLU,  # Example activation function, change as needed
    hidden_units=64,
    padding=0,
    stride=2,
    pooling_size=1,
    out_channels=8,
    use_fc_batchnorm=True,
    use_cnn_batchnorm=True
).to(DEVICE)

# Load model parameters from saved model file
model_path = ['models/model_CNN_97_fanciful-sweep-1', 
              'models/model_CNN_97_twilight-sweep-1',
              'models/model_CNN_97_misty-sweep-1',
              'models/model_CNN_97_devoted-sweep-1']

model.load_state_dict(torch.load(model_path[0]))

# Set the model to evaluation mode
model.eval()

print("-------- model loaded -----------")

# Load data
dataset_val = model.dataset(data_dir=data_dir / "validation", stmf_data_path=DATA_ROOT / STMF_FILENAME, transform=VAL_TRANSFORM)
print(f"dataset_val = {dataset_val}")
test_data_loader = DataLoader(dataset_val, batch_size=500, shuffle=False, num_workers=1)


print("-------- data loaded -----------")

loss_fn = model.loss_fn
with torch.no_grad():
    print("IN with torch.no_grad()")
    print(len(test_data_loader))
    for i, vdata in enumerate(test_data_loader):
        print("IN for ... enumerate(test_data_loader)")
        # print(vdata)
        spectrogram, target = vdata["spectrogram"].to(DEVICE), vdata["target"].to(DEVICE)
        # print(spectrogram)
        print(spectrogram.shape)
        
        start_time = time.time()
        test_outputs = model(spectrogram)
        end_time = time.time()
        test_loss = loss_fn(test_outputs.squeeze(), target)

        print(f"Number of test samples = {spectrogram.shape[0]}")
        inference_time = (end_time - start_time)/spectrogram.shape[0]
        print(f"inference_time = {inference_time}")


test_rmse = test_loss ** 0.5

print("-------- TEST RESULTS ------------")
print()
print(f"test_rmse = {test_rmse}")

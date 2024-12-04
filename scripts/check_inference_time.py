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
TEST_TRANSFORM = transforms.Compose([
            LoadSpectrogram(root_dir=data_dir / "test"),
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
model_path = 'models/model_CNN_97_fanciful-sweep-1'
model.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
model.eval()

print("-------- model loaded -----------")

# Load data
dataset_test = model.dataset(data_dir=data_dir / "test", stmf_data_path=DATA_ROOT / STMF_FILENAME, transform=TEST_TRANSFORM)
test_data_loader = DataLoader(dataset_test, batch_size=500, shuffle=False, num_workers=1)

print("-------- data loaded -----------")


# Variables to store total inference time and count
total_inference_time = 0
num_images = 0

# Loop through the entire dataset
with torch.no_grad():  # Disable gradient computation for inference
    for i, vdata in enumerate(test_data_loader):
        # Extract spectrogram for the current sample
        spectrogram = vdata["spectrogram"].to(DEVICE)
        
        # Measure inference time for this image
        start_time = time.time()

        # Run the model on the current spectrogram
        _ = model(spectrogram)

        end_time = time.time()

        # Calculate inference time for this image
        inference_time = end_time - start_time
        total_inference_time += inference_time
        num_images += 1

        # Print the inference time for each image (for debugging purposes)
        # print(f"Inference time for image {i+1}: {inference_time:.6f} seconds")

# Calculate the mean inference time across all images
mean_inference_time = total_inference_time / num_images if num_images > 0 else 0

# Print the average inference time
print(f"Mean inference time per image: {mean_inference_time:.6f} seconds")
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

# Model parameter of 5 best models
num_conv_layer_val = [3]
num_fc_layers_val = [3]
conv_dropout_val = [0.2]
linear_dropout_val = [0]
kernel_size_val =[(5, 7)]
activation_val = [nn.ReLU]  # Example activation function, change as needed
hidden_units_val = [64]
padding_val = [0]
stride_val = [2]
pooling_size_val = [1]
out_channels_val = [8]
use_fc_batchnorm_val = [True]
use_cnn_batchnorm_val = [True]


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
dataset_test = model.dataset(data_dir=data_dir / "test", stmf_data_path=DATA_ROOT / STMF_FILENAME, transform=TEST_TRANSFORM)
test_data_loader = DataLoader(dataset_test, batch_size=500, shuffle=False, num_workers=1)

print("-------- data loaded -----------")


# Variables to store total inference time and count
total_inference_time = 0
num_images = 0

# Loop through the entire dataset
with torch.no_grad():  # Disable gradient computation for inference
    print(len(test_data_loader))
    for i, vdata in enumerate(test_data_loader):
        print(f"{i} IN enumerate(test_data_loader)")
        print(vdata.shape)
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



# # ------------------ BASELINE MODEL --------------
# class SpectrVelCNNRegr(nn.Module):
#     """Baseline model for regression to the velocity

#     Use this to benchmark your model performance.
#     """

#     loss_fn = mse_loss
#     dataset = SpectrogramDataset

#     def __init__(self):
#         super().__init__()
        
        
#         self.conv1=nn.Sequential(
#             nn.Conv2d(in_channels=6,
#                       out_channels=16,
#                       kernel_size=5,
#                       stride=1,
#                       padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )

#         self.conv2=nn.Sequential(
#             nn.Conv2d(
#                 in_channels=16,
#                 out_channels=32,
#                 kernel_size=5,
#                 stride=1,
#                 padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )

#         self.conv3=nn.Sequential(
#             nn.Conv2d(in_channels=32,
#                       out_channels=64,
#                       kernel_size=5,
#                       stride=1,
#                       padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.conv4=nn.Sequential(
#             nn.Conv2d(in_channels=64,
#                       out_channels=128,
#                       kernel_size=3,
#                       stride=1,
#                       padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.flatten=nn.Flatten()
#         self.linear1=nn.Linear(in_features=37120,out_features=1024)
#         self.linear2=nn.Linear(in_features=1024,out_features=256)
#         self.linear3=nn.Linear(in_features=256,out_features=1)
    
#     def _input_layer(self, input_data):
#         return self.conv1(input_data)

#     def _hidden_layer(self, x):
#         x=self.conv2(x)
#         x=self.conv3(x)
#         x=self.conv4(x)
#         x=self.flatten(x)
#         x=self.linear1(x)
#         return self.linear2(x)

#     def _output_layer(self, x):
#         return self.linear3(x)

#     def forward(self, input_data):
#         x = self._input_layer(input_data)
#         x = self._hidden_layer(x)
#         return self._output_layer(x)



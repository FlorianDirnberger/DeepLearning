import torch
import torch.nn as nn
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from torch.utils.data import DataLoader
from custom_transforms import LoadSpectrogram, NormalizeSpectrogram, ToTensor, InterpolateSpectrogram
from data_management import make_dataset_name
from pathlib import Path
from models import CNN_97, SpectrVelCNNRegr
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

# Model Setup

# Baseline model
model_baseline = SpectrVelCNNRegr()
# model_1 = CNN_97(
#     num_conv_layers=2,  # Example value, use the same as the trained model
#     num_fc_layers=3,
#     conv_dropout=0,
#     linear_dropout=0,
#     kernel_size=(5, 5),
#     activation=nn.ReLU,
#     hidden_units=64,
#     padding=1,
#     stride=1,
#     pooling_size=1,
#     out_channels=8,
#     use_fc_batchnorm=True,
#     use_cnn_batchnorm=True
# ).to(DEVICE)

# Best of approach 3 (CNN optimization)
model_2 = CNN_97(
    num_conv_layers=3,  # Example value, use the same as the trained model
    num_fc_layers=3,
    conv_dropout=0,
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

# Load model parameters
model_paths = ["models/model_SpectrVelCNNRegr_atomic-night-34", "models/model_CNN_97_fanciful-sweep-1"]
model_baseline.load_state_dict(torch.load(model_paths[0]))
model_2.load_state_dict(torch.load(model_paths[1]))

# Set models to evaluation mode
model_baseline.eval()
model_2.eval()

# Load data
dataset_val = model_baseline.dataset(data_dir=data_dir / "validation", stmf_data_path=DATA_ROOT / STMF_FILENAME, transform=VAL_TRANSFORM)
test_data_loader = DataLoader(dataset_val, batch_size=500, shuffle=False, num_workers=1)

# RMSE Calculation
def calculate_rmse(model, data_loader):
    loss_fn = model.loss_fn
    rmse_values = []
    with torch.no_grad():
        for vdata in data_loader:
            spectrogram, target = vdata["spectrogram"].to(DEVICE), vdata["target"].to(DEVICE)
            test_outputs = model(spectrogram)
            test_loss = loss_fn(test_outputs.squeeze(), target)
            rmse_values.extend((test_outputs.squeeze() - target).pow(2).cpu().numpy().tolist())
    return rmse_values

rmse_model_baseline = calculate_rmse(model_baseline, test_data_loader)
rmse_model_2 = calculate_rmse(model_2, test_data_loader)

# Perform Student's t-test
t_stat, p_value = ttest_ind(rmse_model_baseline, rmse_model_2)

# Plot RMSE comparison
plt.figure()
plt.boxplot([rmse_model_baseline, rmse_model_2], tick_labels=["Baseline Model", "Optimized CNN"])
plt.title(f"RMSE Comparison (p-value = {p_value:.3e})")
plt.ylabel("RMSE")
# plt.legend(["Baseline Model", "Optimized CNN"], loc="upper right")
plt.grid(True) 
plt.savefig("rmse_comparison_plot.png")
plt.show()

print("-------- TEST RESULTS ------------")
print(f"RMSE Baseline Model: {rmse_model_baseline}")
print(f"RMSE Optimized CNN: {rmse_model_2}")
print(f"t-statistic: {t_stat}, p-value: {p_value}")

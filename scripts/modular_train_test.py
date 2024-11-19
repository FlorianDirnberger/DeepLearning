from pathlib import Path
from numpy import log10
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import wandb
import torch.nn as nn
from dotenv import load_dotenv
import os

from custom_transforms import LoadSpectrogram, NormalizeSpectrogram, ToTensor, InterpolateSpectrogram
from data_management import make_dataset_name
#from models import SpectrVelCNNRegr, CNN_97, weights_init_uniform_rule
from models import  CNN_97, weights_init_uniform_rule

# GROUP NUMBER
GROUP_NUMBER = 97

# CONSTANTS TO LEAVE
DATA_ROOT = Path(f"/dtu-compute/02456-p4-e24/data") 
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
STMF_FILENAME = "stmf_data_3.csv"
NFFT = 512
TS_CROPTWIDTH = (-150, 200)
VR_CROPTWIDTH = (-60, 15)
DEVICE = "cpu"


# Load environment variables from .env file
load_dotenv()

# Use the API key from the environment variable
wandb_api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=wandb_api_key)


def train_one_epoch(loss_fn, model, train_data_loader, optimizer):
    running_loss = 0.
    total_loss = 0.

    for i, data in enumerate(train_data_loader):
        spectrogram, target = data["spectrogram"].to(DEVICE), data["target"].to(DEVICE)
        
        #print(f"Batch {i} - Spectrogram shape: {spectrogram.shape}, Target shape: {target.shape}")
        
        optimizer.zero_grad()

        outputs = model(spectrogram)

        # Compute the loss and its gradients
        loss = loss_fn(outputs.squeeze(), target)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        total_loss += loss.item()
        if i % train_data_loader.batch_size == train_data_loader.batch_size - 1:
            last_loss = running_loss / train_data_loader.batch_size
            print(f'  batch {i + 1} loss: {last_loss}')
            running_loss = 0.

    return total_loss / (i+1)

def train():
    with wandb.init() as run:
        config = run.config
        #activation_fn = nn.ReLU
        activation_fn_map = {
            'ReLU': nn.ReLU,
            'LeakyReLU': nn.LeakyReLU,
            'Tanh': nn.Tanh,
            'Sigmoid': nn.Sigmoid
        }
        

        

        # Initialize model with the current WandB configuration
        model = CNN_97(
            num_conv_layers = config.num_conv_layers,
            num_fc_layers=config.num_fc_layers,
            conv_dropout=config.conv_dropout,
            linear_dropout=config.linear_dropout,
            kernel_size=config.kernel_size,
            activation=activation_fn_map[config.activation_fn],
            hidden_units=config.hidden_units,
            padding = config.padding,
            stride = config.stride,
            pooling_size = config.pooling_size,
            out_channels = config.out_channels 
        ).to(DEVICE)

        model.apply(weights_init_uniform_rule)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
        loss_fn = model.loss_fn

        # Data Setup
        dataset_name = make_dataset_name(nfft=NFFT, ts_crop_width=TS_CROPTWIDTH, vr_crop_width=VR_CROPTWIDTH)
        data_dir = DATA_ROOT / dataset_name

        TRAIN_TRANSFORM = transforms.Compose([
            LoadSpectrogram(root_dir=data_dir / "train"),
            NormalizeSpectrogram(),
            ToTensor(),
            InterpolateSpectrogram()
        ])
        TEST_TRANSFORM = transforms.Compose([
            LoadSpectrogram(root_dir=data_dir / "test"),
            NormalizeSpectrogram(),
            ToTensor(),
            InterpolateSpectrogram()
        ])

        dataset_train = model.dataset(data_dir=data_dir / "train", stmf_data_path=DATA_ROOT / STMF_FILENAME, transform=TRAIN_TRANSFORM)
        dataset_test = model.dataset(data_dir=data_dir / "test", stmf_data_path=DATA_ROOT / STMF_FILENAME, transform=TEST_TRANSFORM)
        
        train_data_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=4)
        test_data_loader = DataLoader(dataset_test, batch_size=500, shuffle=False, num_workers=1)

        # Training Loop
        best_vloss = float('inf')
        for epoch in range(config.epochs):
            print(f'EPOCH {epoch + 1}:')
            model.train(True)
            avg_loss = train_one_epoch(loss_fn, model, train_data_loader, optimizer)
            
            rmse = avg_loss ** 0.5
            #log_rmse = log10(rmse)

            running_test_loss = 0.
            model.eval()
            with torch.no_grad():
                for i, vdata in enumerate(test_data_loader):
                    spectrogram, target = vdata["spectrogram"].to(DEVICE), vdata["target"].to(DEVICE)
                    test_outputs = model(spectrogram)
                    test_loss = loss_fn(test_outputs.squeeze(), target)
                    running_test_loss += test_loss.item()

            avg_test_loss = running_test_loss / (i + 1)
            test_rmse = avg_test_loss ** 0.5
            #log_test_rmse = torch.log10(test_rmse)

            print(f'LOSS train {avg_loss} ; LOSS test {avg_test_loss}')
            
            # log metrics to wandb
            wandb.log({
                "loss": avg_loss,
                "rmse": rmse,
                #"log_rmse": log_rmse,
                "test_loss": avg_test_loss,
                "test_rmse": test_rmse,
                #"log_test_rmse": log_test_rmse,
            })

            # Track best performance, and save the model's state
            if avg_test_loss < best_vloss:
                best_vloss = avg_test_loss
                model_path = MODEL_DIR / f"model_{model.__class__.__name__}_{wandb.run.name}"
                torch.save(model.state_dict(), model_path)

    # Ensure that each run is properly finished
    #wandb.finish()

# WandB Sweep Configuration
sweep_config = {
    'method': 'random',  # Specifies grid search to try all configurations
    
    'metric': 
        {'name': 'test_rmse', 'goal': 'minimize'}
    ,
    
    'parameters': {
        'conv_dropout': {
            'values': [0, 0.3, 0.5]
        },
        'linear_dropout': {
            'values': [0, 0.3, 0.5]
        },
        'kernel_size': {
            'values': [3, 5, 7]
        },
        'hidden_units': {
            'values': [64, 128, 256] # he used 1028 
        },
        'learning_rate': {
            'values': [1e-4, 1e-5, 1e-6]
        },
        'epochs': {
            'values': [50]
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'num_conv_layers':{
            'values':[1,2,3]
        },
        'num_fc_layers':{
            'values': [1,2,3]
        },
        'stride': {
            'values': [1,2,3]
        },
        'padding': {
            'values': [1,2,3]
        },
        'pooling_size':{
            'values': [1,2,4]
        },
        'out_channels':{
            'values': [16,32] # at least 2^max_num_conv layers #he used 16 
        },
        'activation_fn': {
            'values': ['ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid']
        }
    }
}

if __name__ == "__main__":
    # Initialize the sweep in WandB
    sweep_id = wandb.sweep(sweep_config, project="Deep Learning Project Group 97")
    wandb.agent(sweep_id, train)
import os
from pathlib import Path
from dotenv import load_dotenv
from numpy import log10
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt
from custom_transforms import LoadSpectrogram, NormalizeSpectrogram, ToTensor, InterpolateSpectrogram, Prepro
from data_management import make_dataset_name
from models import CNN_97, weights_init,weights_init_uniform_rule
from typing import Union, Optional
# GROUP NUMBER
GROUP_NUMBER = 97

# CONSTANTS TO MODIFY AS YOU WISH
NUM_WORKERS = 4
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
#DEVICE = "cpu"
# Load environment variables from .env file
load_dotenv()

# Use the API key from the environment variable
wandb_api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=wandb_api_key)

# CONSTANTS TO LEAVE
DATA_ROOT = Path(f"/dtu-compute/02456-p4-e24/data") 
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
STMF_FILENAME = "stmf_data_3.csv"
NFFT = 512
TS_CROPTWIDTH = (-150, 200)
VR_CROPTWIDTH = (-60, 15)




def is_valid_architecture(input_shape, num_conv_layers, kernel_size, stride, padding, pooling_size):
    """
    Validates the compatibility of convolutional and pooling layers with the input dimensions.
    Args:
        input_shape (tuple): Shape of the input data (batch_size, channels, height, width).
        num_conv_layers (int): Number of convolutional layers.
        kernel_size (tuple): Kernel size for convolution.
        stride (int): Stride size for convolution.
        padding (int): Padding size for convolution.
        pooling_size (int): Pooling size.
    Returns:
        bool: True if the architecture is valid, False otherwise.
    """
    _, _, height, width = input_shape
    current_height, current_width = height, width

    for _ in range(num_conv_layers):
        # Compute height and width after convolution
        conv_height = (current_height - kernel_size[0] + 2 * padding) // stride + 1
        conv_width = (current_width - kernel_size[1] + 2 * padding) // stride + 1

        if conv_height <= 0 or conv_width <= 0:
            return False  # Invalid dimensions after convolution

        current_height, current_width = conv_height, conv_width

        # Compute height and width after pooling
        pool_height = current_height // pooling_size
        pool_width = current_width // pooling_size

        if pool_height <= 0 or pool_width <= 0:
            return False  # Invalid dimensions after pooling

        current_height, current_width = pool_height, pool_width

    return True



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
        }
        kernel_size_map = {
            '3x3': (3, 3), 
            '5x5': (5, 5),
            '7x7': (7, 7), 
            '1x3': (1, 3), 
            '3x1': (3, 1), 
            '3x5': (3, 5), 
            '5x3': (5, 3), 
            '3x7': (3, 7), 
            '7x3': (7, 3)
        }


         # Validate architecture before initializing the model
        input_shape = (16, 6, 74, 918)  # [batch_size, channels, height, width]
        if not is_valid_architecture(
            input_shape=input_shape,
            num_conv_layers=config.num_conv_layers,
            kernel_size=kernel_size_map[config.kernel_size],
            stride=config.stride,
            padding=config.padding,
            pooling_size=config.pooling_size
        ):
            print(f"Invalid architecture configuration: {config}")
            return  # Skip this run


        # Initialize model with the current WandB configuration
        model = CNN_97(
            num_conv_layers = config.num_conv_layers,
            num_fc_layers=config.num_fc_layers,
            conv_dropout=config.conv_dropout,
            linear_dropout=config.linear_dropout,
            kernel_size=kernel_size_map[config.kernel_size],
            activation=activation_fn_map[config.activation_fn],
            hidden_units=config.hidden_units,
            padding = config.padding,
            stride = config.stride,
            pooling_size = config.pooling_size,
            out_channels = config.out_channels,
            use_fc_batchnorm = config.use_fc_batchnorm,
            use_cnn_batchnorm = config.use_cnn_batchnorm 
        ).to(DEVICE)

        # Weight initializations depending on activation function
        weights_init_map = {
            'Uniform': lambda m: weights_init_uniform_rule(m),
            'Xavier_uniform': lambda m: weights_init(m, init_type='Xavier_uniform'),
            'Xavier_normal': lambda m: weights_init(m, init_type='Xavier_normal'),
            'Kaiming_uniform': lambda m: weights_init(m, init_type='Kaiming_uniform'),
            'Kaiming_normal': lambda m: weights_init(m, init_type='Kaiming_normal')
        }
        valid_activation_init_combinations = {
            'ReLU': ['Uniform', 'Kaiming_uniform', 'Kaiming_normal'],
            'LeakyReLU': ['Uniform', 'Kaiming_uniform', 'Kaiming_normal'],
            'Tanh': ['Uniform', 'Xavier_uniform', 'Xavier_normal'],
            'Sigmoid': ['Uniform', 'Xavier_uniform', 'Xavier_normal'],
            'Swish': ['Uniform', 'Kaiming_uniform', 'Kaiming_normal'],
            'Mish': ['Uniform', 'Kaiming_uniform', 'Kaiming_normal']
        }
        if config.weights_init not in valid_activation_init_combinations[config.activation_fn]:
            #print(f"Invalid combination: {config.activation_fn} + {config.weights_init}") # debugging
            return # skips the current run

        model.apply(weights_init_map[config.weights_init])


        # Optimizer mapping
        optimizer_map = {
            'SGD': torch.optim.SGD,
            'AdamW': torch.optim.AdamW,
        }

        # Validate weight decay combinations
        valid_optimizer_weight_decay_combinations = {
            'SGD': sweep_config['parameters']['weight_decay']['values'],
            'AdamW': sweep_config['parameters']['weight_decay']['values'],
            'AdaGrad': sweep_config['parameters']['weight_decay']['values'],
            'Adam': sweep_config['parameters']['weight_decay']['values'],
        }

        # Validate momentum combinations (only relevant for SGD)
        valid_optimizer_momentum_combinations = {
            'SGD': sweep_config['parameters']['momentum']['values']
        }

        # Check weight decay for the current optimizer
        if config.weight_decay not in valid_optimizer_weight_decay_combinations.get(config.optimizer, []):
            print(f"Invalid combination: {config.optimizer} + {config.weight_decay}")
            return  # Skip the current run

        # Check momentum only if the optimizer uses it
        if config.optimizer == 'SGD' and config.momentum not in valid_optimizer_momentum_combinations['SGD']:
            print(f"Invalid combination: {config.optimizer} + {config.weight_decay} + {config.momentum}")
            return  # Skip the current run

        # Define optimizer parameters
        if config.optimizer == 'SGD':
            optimizer_params = {
                'params': model.parameters(),
                'lr': config.learning_rate,
                'weight_decay': config.weight_decay,
                'momentum': config.momentum,
            }
        elif config.optimizer in ['AdamW', 'Adam', 'AdaGrad']:
            optimizer_params = {
                'params': model.parameters(),
                'lr': config.learning_rate,
                'weight_decay': config.weight_decay,  # Only for optimizers that support weight decay
            }


        # print(f"Optimizer params for {config.optimizer}: {optimizer_params}")
        optimizer = optimizer_map[config.optimizer](**optimizer_params)

        loss_fn = model.loss_fn

        # Data Setup
        dataset_name = make_dataset_name(nfft=NFFT, ts_crop_width=TS_CROPTWIDTH, vr_crop_width=VR_CROPTWIDTH)
        data_dir = DATA_ROOT / dataset_name

        TRAIN_TRANSFORM = transforms.Compose(
        [LoadSpectrogram(root_dir=data_dir / "train"),
        Prepro(),
        ToTensor(),
        InterpolateSpectrogram()]
        )
        TEST_TRANSFORM = transforms.Compose(
            [LoadSpectrogram(root_dir=data_dir / "test"),
            Prepro(),
            ToTensor(),
            InterpolateSpectrogram()]
        )

        dataset_train = model.dataset(data_dir=data_dir / "train", stmf_data_path=DATA_ROOT / STMF_FILENAME, transform=TRAIN_TRANSFORM)
        dataset_test = model.dataset(data_dir=data_dir / "test", stmf_data_path=DATA_ROOT / STMF_FILENAME, transform=TEST_TRANSFORM)
        
        train_data_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=4)
        test_data_loader = DataLoader(dataset_test, batch_size=500, shuffle=False, num_workers=1)

        # Training Loop
        # Initialize variables to track early stopping
        previous_val_loss = float('inf')  # Start with a very high value
        exceed_val_loss_count = 0  # Counter for consecutive increases in validation loss
        max_increase_epochs = 6  # Number of consecutive epochs allowed for increasing loss
        delta = 0.001  # Minimum improvement needed to reset early stopping counter
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
            validation_rmse = avg_test_loss ** 0.5
            #log_validation_rmse = torch.log10(validation_rmse)

            #total_params = sum(p.numel() for p in model().parameters())

            print(f'LOSS train {avg_loss} ; LOSS test {avg_test_loss}')
            
            # log metrics to wandb
            wandb.log({
                "train_loss": avg_loss,
                "train_rmse": rmse,
                #"log_rmse": log_rmse,
                "validation_loss": avg_test_loss,
                "validation_rmse": validation_rmse,
                #"log_validation_rmse": log_validation_rmse,
                #"total_parameter": total_params
            })
            # Track best performance, and save the model's state
            if avg_test_loss < best_vloss:
                best_vloss = avg_test_loss
                model_path = MODEL_DIR / f"model_{model.__class__.__name__}_{wandb.run.name}"
                torch.save(model.state_dict(), model_path)
                
           # Early Stopping Logic
            if validation_rmse < (previous_val_loss - delta):  # Check improvement with delta
                previous_val_loss = validation_rmse
                exceed_val_loss_count = 0  # Reset counter on improvement
            else:
                exceed_val_loss_count += 1
                if exceed_val_loss_count >= max_increase_epochs:  # Stop if insufficient improvement for too long
                    break

            # Additional conditions to break training based on validation_rmse at specific epochs
            if epoch == 5 and validation_rmse > 11:  
                break
            if epoch == 20 and validation_rmse > 4: 
                break
          
    # Ensure that each run is properly finished
    #wandb.finish()

# WandB Sweep Configuration
sweep_config = {
    'method': 'random',  # Specifies grid search to try all configurations
    
    'metric': 
        {'name': 'validation_rmse', 'goal': 'minimize'}
    ,
    
    'parameters': {
        'conv_dropout': {
            'values': [0,0.1,0.3] #[0.1, 0.3, 0.5]
        },
        'linear_dropout': {
            'values': [0,0.1,0.3] #[0.1, 0.3, 0.5]
        },
        'kernel_size': {
            'values': ['5x5','7x7','3x5','5x3']
        },
        'hidden_units': {
            'values': [64,128,256] #[64, 128, 256]
        },
        'learning_rate': {
            'values': [1e-4]
        },
        'epochs': {
            'values': [40] #[10, 20, 50]
        },
        'batch_size': {
            'values': [16] #[16, 32, 64]
        },
        'num_conv_layers':{
            'values': [3,4]
        },
        'num_fc_layers':{
            'values': [2,3] #[1,2,3]
        },
        'stride': {
            'values': [1,2,3] #[1,2,3]
        },
        'padding': {
            'values': [1,2] #[1,2,3]
        },
        'pooling_size':{
            'values': [1,2] #[1,2,4]
        },
        'out_channels':{
            'values': [8,16] #[16,32] # at least 2^max_num_conv layers
        },
        'activation_fn': {
            'values': ['ReLU','LeakyReLU']
        },
        'weights_init': {
            'values': ['Kaiming_normal']
        },

        # Parameter for batchnorm on cnn_layers
        'use_cnn_batchnorm': {
            'values': [True] #[True, False]
        },

        # Parameter for batchnorm on fc_layers
        'use_fc_batchnorm': {
            'values': [True] #[True, False]
        },
        # Parameters for optimizer
        'optimizer': {
            'values': ['SGD'] #['SGD', 'Adam', 'AdamW', 'AdaGrad']
        },
        'weight_decay': {
            'values': [1e-2,1e-5] #[0, 1e-5, 1e-4, 1e-3, 1e-2]        
        },
        'momentum': {
            'values': [0.8,0.9]        
        }
    }
}

if __name__ == "__main__":
    # Initialize the sweep in WabndB
    sweep_id = wandb.sweep(sweep_config, project="Preprocess with sobel")
    wandb.agent(sweep_id, train,count=200)





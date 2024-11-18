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
from models import CNN_97, weights_init_uniform_rule, weights_init

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
#DEVICE = "cpu"
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


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
            'Sigmoid': nn.Sigmoid,
            'Swish': nn.SiLU,
            'Mish': nn.Mish
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
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'AdaGrad': torch.optim.Adagrad
        }

        # To avoid redundant runs
        valid_optimizer_weight_decay_combinations = {
            'SGD': sweep_config['parameters']['weight_decay']['values'],
            'Adam': [sweep_config['parameters']['weight_decay']['values'][0]],
            'AdamW': sweep_config['parameters']['weight_decay']['values'],
            'AdaGrad': [sweep_config['parameters']['weight_decay']['values'][0]]
        }

        valid_optimizer_momentum_combinations = {
            'SGD': sweep_config['parameters']['momentum']['values'],
            'Adam': [sweep_config['parameters']['momentum']['values'][0]],
            'AdamW': [sweep_config['parameters']['momentum']['values'][0]],
            'AdaGrad': [sweep_config['parameters']['momentum']['values'][0]]
        }

        if config.weight_decay not in valid_optimizer_weight_decay_combinations[config.optimizer]:
            print(f"Invalid combination: {config.optimizer} + {config.weight_decay}")
            return # skips the current run
        
        if config.momentum not in valid_optimizer_momentum_combinations[config.optimizer]:
            print(f"Invalid combination: {config.optimizer} + {config.weight_decay} + {config.momentum}")
            return # skips the current run

        if config.optimizer == 'SGD':
            optimizer_params = {
                'params': model.parameters(),
                'lr': config.learning_rate,
                'weight_decay': config.weight_decay,
                'momentum': config.momentum 
            }
        
        elif config.optimizer == 'AdamW':
            optimizer_params = {
                'params': model.parameters(),
                'lr': config.learning_rate,
                'weight_decay': config.weight_decay  # Only for AdamW
            }

        else:  # For Adam and AdaGrad
            optimizer_params = {
                'params': model.parameters(),
                'lr': config.learning_rate          # Basic parameters for Adam and AdaGrad
            }


        # print(f"Optimizer params for {config.optimizer}: {optimizer_params}")
        optimizer = optimizer_map[config.optimizer](**optimizer_params)

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

            total_params = sum(p.numel() for p in model().parameters())

            print(f'LOSS train {avg_loss} ; LOSS test {avg_test_loss}')
            
            # log metrics to wandb
            wandb.log({
                "train_loss": avg_loss,
                "train_rmse": rmse,
                #"log_rmse": log_rmse,
                "validation_loss": avg_test_loss,
                "validation_rmse": test_rmse,
                #"log_test_rmse": log_test_rmse,
                "total_parameter": total_params
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
    'method': 'grid',  # Specifies grid search to try all configurations
    
    'metric': 
        {'name': 'test_rmse', 'goal': 'minimize'}
    ,
    
    'parameters': {
        'conv_dropout': {
            'values': [0, 0.3] #[0.1, 0.3, 0.5]
        },
        'linear_dropout': {
            'values': [0] #[0.1, 0.3, 0.5]
        },
        'kernel_size': {
            'values': ['3x3']
        },
        'hidden_units': {
            'values': [32, 128] #[64, 128, 256]
        },
        'learning_rate': {
            'values': [1e-4, 1e-5]
        },
        'epochs': {
            'values': [50] #[10, 20, 50]
        },
        'batch_size': {
            'values': [32, 64] #[16, 32, 64]
        },
        'num_conv_layers':{
            'values': [2,3]
        },
        'num_fc_layers':{
            'values': [2] #[1,2,3]
        },
        'stride': {
            'values': [1] #[1,2,3]
        },
        'padding': {
            'values': [1] #[1,2,3]
        },
        'pooling_size':{
            'values': [2] #[1,2,4]
        },
        'out_channels':{
            'values': [16,32] #[16,32] # at least 2^max_num_conv layers
        },
        'activation_fn': {
            'values': ['ReLU']
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
            'values': ['SGD', 'AdamW'] #['SGD', 'Adam', 'AdamW', 'AdaGrad']
        },
        'weight_decay': {
            'values': [0, 1e-5] #[0, 1e-5, 1e-4, 1e-3, 1e-2]        
        },
        'momentum': {
            'values': [0.9] #[0.7, 0.8, 0.9]        
        }
    }
}

if __name__ == "__main__":
    # Initialize the sweep in WandB
    sweep_id = wandb.sweep(sweep_config, project="Deep Learning Project Group 97")
    wandb.agent(sweep_id, train)

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
from custom_transforms import LoadSpectrogram, NormalizeSpectrogram, ToTensor, InterpolateSpectrogram, Prepro
from data_management import make_dataset_name
from models import SpectrVelCNNRegr, weights_init_uniform_rule
from typing import Union, Optional
# GROUP NUMBER
GROUP_NUMBER = 97

# CONSTANTS TO MODIFY AS YOU WISH
MODEL = SpectrVelCNNRegr
LEARNING_RATE = 10**-5
EPOCHS = 300 # the model converges in test perfermance after ~250-300 epochs
BATCH_SIZE = 10
NUM_WORKERS = 4
OPTIMIZER = torch.optim.SGD
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


# You can set the model path name in case you want to keep training it.
# During the training/testing loop, the model state is saved
# (only the best model so far is saved)
LOAD_MODEL_FNAME = None
# LOAD_MODEL_FNAME = f"model_{MODEL.__name__}_bright-candle-20"

# CONSTANTS TO LEAVE
DATA_ROOT = Path(f"/dtu-compute/02456-p4-e24/data") 
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
STMF_FILENAME = "stmf_data_3.csv"
NFFT = 512
TS_CROPTWIDTH = (-150, 200)
VR_CROPTWIDTH = (-60, 15)

# Define the save directory using Path and expand the user home directory
#save_dir = Path('~/my_project_dir/DeepLearning97/outputs').expanduser()

# Create the directory if it doesn't exist
#save_dir.mkdir(parents=True, exist_ok=True)

def plot_spectrogram_with_annotations(
    spectrogram: Union[Tensor, np.ndarray],
    spectrogram_channel: int = 0,
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """Plot spectrogram for the specified channel.

    Args:
        spectrogram (Union[Tensor, np.ndarray]): Input spectrogram to plot.
        spectrogram_channel (int, optional): Spectrogram channel to plot. Defaults to 0.
        save_path (Optional[Union[str, Path]], optional): Path to save the plot image. If None, displays the plot.
    """
    # If the spectrogram is a tensor, convert it to numpy array
    if isinstance(spectrogram, Tensor):
        spectrogram = spectrogram.cpu().numpy()
    elif not isinstance(spectrogram, np.ndarray):
        raise TypeError("spectrogram must be a Tensor or ndarray")

    # Select the specified channel
    spectrogram_channel_data = spectrogram[spectrogram_channel, :, :]

    # Use the spectrogram data for plotting
    spectrogram_to_plot = spectrogram_channel_data

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.imshow(
        spectrogram_to_plot,
        aspect="auto",
        origin="lower",
        cmap="jet"
    )
    plt.colorbar(label='Amplitude')
    plt.xlabel("Time [s]")
    plt.ylabel("Radial Velocity [m/s]")
    plt.title(f"Spectrogram - Channel {spectrogram_channel}")

    # Save or display the plot
    if save_path:
        save_path = Path(save_path).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to {save_path}")
    else:
        plt.show()



def train_one_epoch(loss_fn, model, train_data_loader):
    running_loss = 0.
    last_loss = 0.
    total_loss = 0.

    for i, data in enumerate(train_data_loader):
        spectrogram, target = data["spectrogram"].to(DEVICE), data["target"].to(DEVICE)

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
            last_loss = running_loss / train_data_loader.batch_size # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return total_loss / (i+1)

if __name__ == "__main__":
    print(f"Using {DEVICE} device")

    # DATA SET SETUP
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
    dataset_train = MODEL.dataset(data_dir= data_dir / "train",
                           stmf_data_path = DATA_ROOT / STMF_FILENAME,
                           transform=TRAIN_TRANSFORM)

    dataset_test = MODEL.dataset(data_dir= data_dir / "test",
                           stmf_data_path = DATA_ROOT / STMF_FILENAME,
                           transform=TEST_TRANSFORM)
    
    train_data_loader = DataLoader(dataset_train, 
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=NUM_WORKERS)
    test_data_loader = DataLoader(dataset_test,
                                  batch_size=500,
                                  shuffle=False,
                                  num_workers=1)
    


    if LOAD_MODEL_FNAME is not None:
        model = MODEL().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_DIR / LOAD_MODEL_FNAME))
        model.eval()
    else:
        model = MODEL().to(DEVICE)
        model.apply(weights_init_uniform_rule)

    optimizer = OPTIMIZER(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    
    
    # Set up wandb for reporting
    wandb.init(
        project=f"02456_group_{GROUP_NUMBER}",
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": MODEL.__name__,
            "dataset": MODEL.dataset.__name__,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "transform": "|".join([str(tr).split(".")[1].split(" ")[0] for tr in dataset_train.transform.transforms]),
            "optimizer": OPTIMIZER.__name__,
            "loss_fn": model.loss_fn.__name__,
            "nfft": NFFT
        }
    )

    # Define model output to save weights during training
    MODEL_DIR.mkdir(exist_ok=True)
    model_name = f"model_{MODEL.__name__}_{wandb.run.name}"
    model_path = MODEL_DIR / model_name


    ## TRAINING LOOP
    epoch_number = 0
    best_vloss = 1_000_000.

    # import pdb; pdb.set_trace()

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on
        model.train(True)

        # Do a pass over the training data and get the average training MSE loss
        avg_loss = train_one_epoch(MODEL.loss_fn, model, train_data_loader)
        
        # Calculate the root mean squared error: This gives
        # us the opportunity to evaluate the loss as an error
        # in natural units of the ball velocity (m/s)
        rmse = avg_loss**(1/2)

        # Take the log as well for easier tracking of the
        # development of the loss.
        log_rmse = log10(rmse)

        # Reset test loss
        running_test_loss = 0.

        # Set the model to evaluation mode
        model.eval()

        # Disable gradient computation and evaluate the test data
        with torch.no_grad():
            for i, vdata in enumerate(test_data_loader):
                # Get data and targets
                spectrogram, target = vdata["spectrogram"].to(DEVICE), vdata["target"].to(DEVICE)
                
                # Get model outputs
                test_outputs = model(spectrogram)

                # Calculate the loss
                test_loss = MODEL.loss_fn(test_outputs.squeeze(), target)

                # Add loss to runnings loss
                running_test_loss += test_loss

        # Calculate average test loss
        avg_test_loss = running_test_loss / (i + 1)

        # Calculate the RSE for the training predictions
        test_rmse = avg_test_loss**(1/2)

        # Take the log as well for visualisation
        log_test_rmse = torch.log10(test_rmse)

        print('LOSS train {} ; LOSS test {}'.format(avg_loss, avg_test_loss))
        
        # log metrics to wandb
        wandb.log({
            "loss": avg_loss,
            "rmse": rmse,
            "log_rmse": log_rmse,
            "test_loss": avg_test_loss,
            "Validation_rmse": test_rmse,
            "log_test_rmse": log_test_rmse,
        })

        # Track best performance, and save the model's state
        if avg_test_loss < best_vloss:
            best_vloss = avg_test_loss
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

    wandb.finish()





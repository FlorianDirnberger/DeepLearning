# plot_observations.py

from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import Tensor
from typing import Union, Optional

# No need to import ToTensor if using _to_tensor function
# from custom_transforms import ToTensor

# Constants
TS_CROPTWIDTH = (-150, 200)  # Time crop width in milliseconds
VR_CROPTWIDTH = (-60, 15)    # Radial velocity crop width in m/s

def _to_tensor(spectrogram):
    """Convert a NumPy ndarray spectrogram to a PyTorch Tensor."""
    # Swap channel axis because:
    # NumPy image: H x W x C
    # PyTorch image: C x H x W
    spectrogram = spectrogram.transpose((2, 0, 1))
    return torch.from_numpy(spectrogram.astype(np.float32))

def plot_spectrogram_with_annotations(
    spectrogram: Union[Tensor, str, Path, np.ndarray],
    target_vr: float,
    estimated_vr: Optional[float] = None,
    spectrogram_channel: int = 0,
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """Plot spectrogram with true radial velocity annotation.

    Args:
        spectrogram (Union[Tensor, str, Path, np.ndarray]): Input spectrogram to plot.
        target_vr (float): True radial ball velocity.
        estimated_vr (Optional[float], optional): Predicted radial ball velocity. If None, it won't be plotted.
        spectrogram_channel (int, optional): Spectrogram channel to plot (0-5). Defaults to 0.
        save_path (Optional[Union[str, Path]], optional): Path to save the plot image. If None, displays the plot.
    """

    # Load spectrogram if it's a file path
    if isinstance(spectrogram, (str, Path)):
        spectrogram = Path(spectrogram)
        spectrogram = np.load(spectrogram)

    if isinstance(spectrogram, np.ndarray):
        spectrogram = _to_tensor(spectrogram)

    if isinstance(spectrogram, Tensor):
        spectrogram = spectrogram.squeeze()

    # Select the specified channel
    spectrogram = spectrogram[spectrogram_channel, :, :]

    # Set color scale limits based on channel
    if 0 <= spectrogram_channel <= 3:
        vmin = -110
        vmax = -40
    elif 4 <= spectrogram_channel <= 5:
        vmin = -np.pi
        vmax = np.pi
    else:
        raise IndexError("Channel number must be between 0 and 5")

    # Create the plot
    _, ax = plt.subplots(1, 1, figsize=(10, 6))
    img = ax.imshow(
        spectrogram,
        aspect="auto",
        extent=[
            TS_CROPTWIDTH[0] / 1000, TS_CROPTWIDTH[1] / 1000,
            VR_CROPTWIDTH[0], VR_CROPTWIDTH[1]
        ],
        vmin=vmin, vmax=vmax,
        origin="lower",
        interpolation='nearest',
        cmap="jet"
    )
    ax.set_ylabel("Radial Velocity [m/s]")
    ax.set_xlabel("Time [s]")

    # Plot true radial velocity
    ax.plot(
        [TS_CROPTWIDTH[0] / 1000, TS_CROPTWIDTH[1] / 1000],
        [target_vr, target_vr],
        'w--', label=r"True $v_{r}$"
    )

    # Plot estimated radial velocity if provided
    if estimated_vr is not None:
        ax.plot(
            [TS_CROPTWIDTH[0] / 1000, TS_CROPTWIDTH[1] / 1000],
            [estimated_vr, estimated_vr],
            'w:', label=r"Pred. $\bar{v}_{r}$"
        )

    ax.legend()

    # Add colorbar
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label('Power' if spectrogram_channel < 4 else 'Phase')

    # Save or display the plot
    if save_path:
        # Expand user '~' and ensure the directory exists
        save_path = Path(save_path).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Save the plot
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

# Main execution
if __name__ == "__main__":
    # Define start and end observation numbers
    start_obs_no = 136188  # Replace with your starting observation number
    end_obs_no = 136188    # Replace with your ending observation number

    # Load targets CSV to get the true radial velocities
    stmf_data = Path(f"/dtu-compute/02456-p4-e24/data/stmf_data_3.csv")
    targets = pd.read_csv(stmf_data)

    # Set the observation numbers as the index if not already
    if 'ObsNo' in targets.columns:
        targets.set_index('ObsNo', inplace=True)
    elif 'ObservationNumber' in targets.columns:
        targets.set_index('ObservationNumber', inplace=True)
    else:
        # If the index is already observation numbers, skip setting the index
        pass

    # Ensure the index is integer type
    targets.index = targets.index.astype(int)

    # Save directory for the plots
    save_dir = Path("~/my_project_dir/DeepLearning97/outputs").expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the observation number
    obs_no = start_obs_no

    # Loop from start_obs_no to end_obs_no
    while obs_no <= end_obs_no:
        try:
            # Check if spectrogram file exists
            fname = f"/dtu-compute/02456-p4-e24/data/data_fft-512_tscropwidth-150-200_vrcropwidth-60-15/train/{obs_no}_stacked_spectrograms.npy"
            fname = Path(fname)

            if not fname.is_file():
                print(f"Spectrogram file not found for observation number {obs_no}. Skipping.")
                obs_no += 1
                continue

            # Get true radial velocity using the DataFrame index
            if obs_no in targets.index:
                vr = targets.loc[obs_no, 'BallVr']
            else:
                print(f"No data found for observation number {obs_no}. Skipping.")
                obs_no += 1
                continue

            # Estimated radial velocity (optional, set to None if not available)
            estimated_vr = None  # Replace with your estimated radial velocity if available

            # Save path for the plot
            save_path = save_dir / f"spectrogram_{obs_no}.png"

            # Call the plotting function
            plot_spectrogram_with_annotations(
                spectrogram=fname,
                target_vr=vr,
                estimated_vr=estimated_vr,
                spectrogram_channel=4,  # You can change this to channels 0-5
                save_path=save_path     # Provide the save path here
            )
        except Exception as e:
            print(f"Error processing observation number {obs_no}: {e}")
        finally:
            # Increment the observation number
            obs_no += 1

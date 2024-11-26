# contour area
# contour location ( time amplitude)
# oriantition
# shape ?
# 
#
#
#


from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import Tensor
from typing import Union, Optional
import cv2  # OpenCV import
import math
from plot_observations import _to_tensor
from Features import extract_spectrogram_features
# Constants
TS_CROPTWIDTH = (-150, 80)  # Time crop width in milliseconds
VR_CROPTWIDTH = (-60, 0)    # Radial velocity crop width in m/s


def preprocess_spectrogram(spectrogram_np):
    # Step 1: Normalize the spectrogram
    spectrogram_normalized = cv2.normalize(spectrogram_np, None, 0, 1, cv2.NORM_MINMAX)

    # Step 2: Compute the mean and standard deviation
    mean_value = np.mean(spectrogram_normalized)
    std_value = np.std(spectrogram_normalized)

    # Step 3: Define the target range based on 60% of the maximum and one standard deviation
    max_value = np.max(spectrogram_normalized)
    target_value = max_value * 0.70  # 60% of the maximum value
    lower_bound = target_value + std_value  # Lower limit
    # Step 4: Replace values outside the range with zero
    spectrogram_thresholded = np.where(
        (spectrogram_normalized >= lower_bound),
        spectrogram_normalized,
        0
    )

    # Step 5: Apply Gaussian Blur for noise reduction
    smoothed_spectrogram = cv2.GaussianBlur(spectrogram_thresholded, (3, 3), sigmaX=1)

    return smoothed_spectrogram

def find_and_draw_contours(spectrogram_np):
    """
    Process a spectrogram, find contours, draw them on the spectrogram, and mark their centers of mass.
    Only draws contours that are at least 20% the size of the largest contour.

    Args:
        spectrogram_np (np.ndarray): Input spectrogram as a NumPy array.

    Returns:
        np.ndarray: Image with contours and centers of mass drawn.
    """

    # Step 2: Threshold the spectrogram to create a binary image
    _, thresholded = cv2.threshold((spectrogram_np * 255).astype(np.uint8), 20, 255, cv2.THRESH_BINARY)

    # Step 3: Find contours in the binary image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4: Find the largest contour area
    largest_contour_area = max(cv2.contourArea(contour) for contour in contours) if contours else 0
    area_threshold = largest_contour_area * 0.15

    # Step 5: Convert the normalized spectrogram back to a 3-channel image for visualization
    spectrogram_vis = cv2.cvtColor((spectrogram_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Step 6: Draw the contours and their centers of mass
    for contour in contours:
        # Filter out contours smaller than 20% of the largest contour
        contour_area = cv2.contourArea(contour)
        if contour_area < area_threshold:
            continue

        # Draw the contour
        cv2.drawContours(spectrogram_vis, [contour], -1, (0, 0, 255), 1)  # Red contour

        # Calculate moments and draw the centroid
        moment = cv2.moments(contour)
        if moment["m00"] != 0:  # Avoid division by zero
            cx = int(moment["m10"] / moment["m00"])
            cy = int(moment["m01"] / moment["m00"])
            cv2.circle(spectrogram_vis, (cx, cy), radius=2, color=(255, 0, 0), thickness=-1)  # Blue circle

    return spectrogram_vis



def Make_spectrogram(
    spectrogram: Union[Tensor, str, Path, np.ndarray],
    spectrogram_channel: int = 0,
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

    # Convert the spectrogram to a NumPy array
    spectrogram_np = spectrogram.numpy()
    return spectrogram_np


def plot_spectrogram_with_annotations(
    spectrogram,
    target_vr: float,
    spectrogram_channel: int = 0,
    save_path: Optional[Union[str, Path]] = None
) -> None:

    vmin = 1
    vmax = 0

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
    save_dir = Path("~/my_project_dir/DeepLearning97/ANNF").expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the observation number
    obs_no = start_obs_no

    # Loop from start_obs_no to end_obs_no
    while obs_no <= end_obs_no:
        try:       
            # Construct file path
            fname = f"/dtu-compute/02456-p4-e24/data/data_fft-512_tscropwidth-150-200_vrcropwidth-60-15/train/{obs_no}_stacked_spectrograms.npy"
            fname = Path(fname)

            if not fname.is_file():
                #print(f"Spectrogram file not found for observation number {obs_no}. Skipping.")
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
            save_path = save_dir / f"spectrogram_filterd_{obs_no}.png"

            spectrogram_channel = 0

            # Call the plotting function
            spec= Make_spectrogram(
                spectrogram=fname,
                spectrogram_channel=spectrogram_channel,  # You can change this to channels 0-5
            )

            worked_spec=preprocess_spectrogram(spec) # pre processing
            print(extract_spectrogram_features(worked_spec))
            con = find_and_draw_contours(worked_spec)
            plot_spectrogram_with_annotations(con,
                                              target_vr=vr,
                                              spectrogram_channel=spectrogram_channel,
                                              save_path=save_path 
                                              )

        except Exception as e:
            print(f"Error processing observation number {obs_no}: {e}")
        finally:
            # Increment the observation number
            obs_no += 1

from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import Tensor
from typing import Union, Optional
import cv2  # OpenCV import
import math


# Constants
TS_CROPTWIDTH = (-150, 200)
VR_CROPTWIDTH = (-60, 15)

def draw_contour_orientation(image, contour, moment, color=(255, 0, 0), thickness=2):
    """Draw the principal axis of the contour based on its orientation.

    Args:
        image (np.ndarray): Image to draw on.
        contour (np.ndarray): Contour points.
        moment (dict): Moments of the contour.
        color (tuple): Line color for visualization.
        thickness (int): Thickness of the line.
    """
    if moment['m00'] == 0:
        return  # Skip contours with no area

    # Compute centroid
    cx = int(moment['m10'] / moment['m00'])
    cy = int(moment['m01'] / moment['m00'])

    # Compute orientation angle in radians
    mu11 = moment['mu11']
    mu20 = moment['mu20']
    mu02 = moment['mu02']

    if mu20 == mu02:  # Avoid division by zero
        return

    angle = 0.5 * math.atan2(2 * mu11, mu20 - mu02)

    # Compute start and end points of the line
    line_length = 50  # Length of the orientation line
    x1 = int(cx + line_length * math.cos(angle))
    y1 = int(cy + line_length * math.sin(angle))
    x2 = int(cx - line_length * math.cos(angle))
    y2 = int(cy - line_length * math.sin(angle))

    # Draw the line on the image
    cv2.line(image, (x1, y1), (x2, y2), color, thickness)

def visualize_contour_orientations(canny_edges, gradient_magnitude_thresholded):
    """Visualize the orientation of contours, filtering out small contours and checking for north orientation.
       Also highlights the center of mass of the leftmost contour.

    Args:
        canny_edges (np.ndarray): Edge-detected image.
        gradient_magnitude_thresholded (np.ndarray): Gradient image for visualization.

    Returns:
        np.ndarray: Image with drawn orientations.
    """
    # Find contours
    contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Compute the largest contour area
    largest_contour_area = max(cv2.contourArea(contour) for contour in contours)
    area_threshold = largest_contour_area * 0.3

    # Filter contours by area
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= area_threshold]

    # Create a copy of the image to draw debug visuals
    debug_image = cv2.cvtColor((gradient_magnitude_thresholded * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Variable to store the leftmost contour and its centroid
    leftmost_centroid = None
    leftmost_x = float('inf')

    # Process contours and draw orientations if they are "north"
    for contour in filtered_contours:
        moment = cv2.moments(contour)
        if moment['m00'] != 0:  # Avoid division by zero for empty contours
            if is_contour_north(moment, tolerance_degrees=30):  # Check if contour is "north"
                # Draw the contour orientation
                draw_contour_orientation(debug_image, contour, moment, color=(0, 255, 0), thickness=1)

                # Compute centroid
                cx = int(moment['m10'] / moment['m00'])
                cy = int(moment['m01'] / moment['m00'])

                # Update leftmost contour
                if cx < leftmost_x:
                    leftmost_x = cx
                    leftmost_centroid = (cx, cy)

    # Draw the center of mass of the leftmost contour
    if leftmost_centroid:
        cv2.circle(debug_image, leftmost_centroid, radius=5, color=(0, 0, 255), thickness=-1)

    return debug_image


def is_contour_north(moment, tolerance_degrees=30):
    """Determine if a contour is oriented north (aligned with y-axis).

    Args:
        moment (dict): Image moments for the contour.
        tolerance_degrees (float): Tolerance in degrees for deviation from the y-axis.

    Returns:
        bool: True if contour is oriented north, False otherwise.
    """
    # Compute orientation angle from moments
    mu11 = moment['mu11']
    mu20 = moment['mu20']
    mu02 = moment['mu02']

    # Avoid division by zero
    if mu20 == mu02:
        return False

    # Compute orientation in radians
    theta = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
    
    # Convert angle to degrees
    angle_degrees = math.degrees(theta)
    #print(angle_degrees)

    # Check if the angle aligns with the y-axis (near 90°)
    return abs(angle_degrees) >= tolerance_degrees


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
    spectrogram_channel_data = spectrogram[spectrogram_channel, :, :]

    # Convert the spectrogram to a NumPy array
    spectrogram_np = spectrogram_channel_data.numpy()

    if spectrogram_np.dtype != np.float64:
        spectrogram_np = spectrogram_np.astype(np.float64)	
   
    sobelx = cv2.Sobel(spectrogram_np, cv2.CV_64F, dx=1, dy=0, ksize=21)
    # Compute the gradient magnitude
    gradient_magnitude = np.sqrt(sobelx**2)

    pectrogram_normalized = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)

  

    # Set color scale limits based on processed data
    # vmin = 0
    # vmax = 1
    vmin = 0
    vmax = 1
    # Create the plot
    _, ax = plt.subplots(1, 1, figsize=(10, 6))
    img = ax.imshow(
        pectrogram_normalized,
        aspect="auto",
        extent=[
            TS_CROPTWIDTH[0] / 1000, TS_CROPTWIDTH[1] / 1000,
            VR_CROPTWIDTH[0], VR_CROPTWIDTH[1]
        ],
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        interpolation='nearest',
        cmap="jet"
    )
    ax.set_ylabel("Radial Velocity [m/s]")
    ax.set_xlabel("Time [s]")

    #Plot true radial velocity
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
    cbar.set_label('Power')

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

    stmf_data = Path(f"/dtu-compute/02456-p4-e24/data/stmf_data_3.csv") 

    # Define start and end observation numbers
    start_obs_no = 136188  # Replace with your starting observation number
    end_obs_no = 137000    # Replace with your ending observation number

    # Load targets CSV to get the true radial velocities
    targets_csv_path = Path(__file__).parent.parent.parent / "data" / "stmf_data_3.csv"
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
    save_dir = Path("~/my_project_dir/DeepLearning97/Threshold_img").expanduser()
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
            save_path = save_dir / f"spectrogram_filterd_sobel_{obs_no}.png"

            # Call the plotting function
            plot_spectrogram_with_annotations(
                spectrogram=fname,
                target_vr=vr,
                estimated_vr=estimated_vr,
                spectrogram_channel=0,  # You can change this to channels 0-5
                save_path=save_path     # Provide the save path here
            )
        except Exception as e:
            print(f"Error processing observation number {obs_no}: {e}")
        finally:
            # Increment the observation number
            obs_no += 1
